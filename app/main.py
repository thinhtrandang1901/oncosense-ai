
from pathlib import Path
import json
import math
import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from sklearn.datasets import load_breast_cancer


# =========================================================
# 1. DUONG DAN DU AN
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "breast_cancer_svm.joblib"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"

TEMPLATE_DIR = ROOT / "app" / "templates"
STATIC_DIR = ROOT / "app" / "static"


# =========================================================
# 2. KHOI TAO FASTAPI
# =========================================================

app = FastAPI(
    title="OncoSense AI",
    description=(
        "Du an hoc tap phan loai du lieu "
        "Breast Cancer Wisconsin bang mo hinh SVM."
    ),
    version="1.0.0",
)


# =========================================================
# 3. CAU HINH SESSION
# =========================================================

SESSION_SECRET = os.getenv("SESSION_SECRET")

if not SESSION_SECRET:
    if os.getenv("RENDER"):
        raise RuntimeError(
            "Thieu SESSION_SECRET. "
            "Hay them bien nay trong Render Environment."
        )

    # Chi dung khi chay thu tren may ca nhan.
    SESSION_SECRET = "local-development-only-change-this-secret"

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=bool(os.getenv("RENDER")),
    max_age=60 * 60 * 8,
)


# =========================================================
# 4. CAU HINH GOOGLE OAUTH
# =========================================================

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

oauth = OAuth()

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url=(
            "https://accounts.google.com/"
            ".well-known/openid-configuration"
        ),
        client_kwargs={
            "scope": "openid email profile",
        },
    )


# =========================================================
# 5. DOC MO HINH SVM VA METADATA
# =========================================================

model = None
metadata = {}

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)


# =========================================================
# 6. DU LIEU BREAST CANCER
# =========================================================

dataset = load_breast_cancer()
FEATURE_NAMES = list(dataset.feature_names)

SAMPLE_FEATURES = {
    name: float(value)
    for name, value in zip(FEATURE_NAMES, dataset.data[0])
}


# =========================================================
# 7. CAU HINH GIAO DIEN
# =========================================================

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )


# =========================================================
# 8. MODEL DU LIEU API
# =========================================================

class PredictionRequest(BaseModel):
    features: dict[str, float]


# =========================================================
# 9. HAM KIEM TRA DANG NHAP
# =========================================================

def is_logged_in(request: Request) -> bool:
    return bool(request.session.get("user"))


def require_login(request: Request):
    if not is_logged_in(request):
        raise HTTPException(
            status_code=401,
            detail="Vui long dang nhap truoc khi su dung.",
        )


# =========================================================
# 10. TRANG DANG NHAP
# =========================================================

@app.get("/login")
async def login_page(request: Request):
    """
    Hien thi trang login.html co nut dang nhap Google.
    """
    if is_logged_in(request):
        return RedirectResponse("/", status_code=303)

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "oauth_configured": False,
                "error": (
                    "Google OAuth chua duoc cau hinh. "
                    "Vui long kiem tra Render Environment."
                ),
            },
            status_code=503,
        )

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "oauth_configured": True,
            "error": None,
        },
    )


# =========================================================
# 11. BAT DAU DANG NHAP GOOGLE
# =========================================================

@app.get("/auth/google")
async def google_login(request: Request):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Google OAuth chua duoc cau hinh tren may chu.",
        )

    redirect_uri = str(request.url_for("auth_callback"))

    return await oauth.google.authorize_redirect(
        request,
        redirect_uri,
    )


# Tuong thich voi duong dan /login cu neu frontend dang dung.
@app.get("/login/google")
async def google_login_legacy(request: Request):
    return await google_login(request)


# =========================================================
# 12. NHAN KET QUA XAC THUC TU GOOGLE
# =========================================================

@app.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Google OAuth chua duoc cau hinh tren may chu.",
        )

    try:
        token = await oauth.google.authorize_access_token(request)

        user = token.get("userinfo")

        if not user:
            user = await oauth.google.parse_id_token(request, token)

        if not user or not user.get("email"):
            raise HTTPException(
                status_code=400,
                detail="Khong lay duoc thong tin tai khoan Google.",
            )

        request.session.clear()
        request.session["user"] = {
            "email": user.get("email"),
            "name": user.get("name", ""),
            "picture": user.get("picture", ""),
        }

        return RedirectResponse("/", status_code=303)

    except HTTPException:
        raise
    except Exception:
        # Khong tra thong tin token/secret ra cho nguoi dung.
        return RedirectResponse(
            "/login?error=google_login_failed",
            status_code=303,
        )


# =========================================================
# 13. DANG XUAT
# =========================================================

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# =========================================================
# 14. KIEM TRA TRANG THAI MAY CHU
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "google_oauth_configured": bool(
            GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET
        ),
    }


# =========================================================
# 15. TRANG CHU - YEU CAU DANG NHAP
# =========================================================

@app.get("/")
async def home(request: Request):
    if not is_logged_in(request):
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "project_name": "OncoSense AI",
            "user": request.session["user"],
        },
    )


# =========================================================
# 16. API LAY DU LIEU MAU
# =========================================================

@app.get("/sample")
async def get_sample(request: Request):
    require_login(request)
    return {"features": SAMPLE_FEATURES}


# =========================================================
# 17. API METADATA
# =========================================================

@app.get("/metadata")
async def get_metadata(request: Request):
    require_login(request)

    return {
        "project": "OncoSense AI",
        "model_name": "Support Vector Machine (SVM)",
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "metrics": metadata.get("metrics", {}),
        "disclaimer": (
            "Chi phuc vu hoc tap, khong dung de chan doan y khoa."
        ),
    }


# =========================================================
# 18. API DU DOAN SVM
# =========================================================

@app.post("/predict")
async def predict(
    request: Request,
    payload: PredictionRequest,
):
    require_login(request)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Chua co mo hinh. Hay kiem tra thu muc artifacts.",
        )

    features = payload.features

    missing = [
        name for name in FEATURE_NAMES
        if name not in features
    ]
    extra = [
        name for name in features
        if name not in FEATURE_NAMES
    ]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Danh sach dac trung chua hop le.",
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    values = [features[name] for name in FEATURE_NAMES]

    if not all(math.isfinite(value) for value in values):
        raise HTTPException(
            status_code=422,
            detail="Cac gia tri dau vao phai la so huu han.",
        )

    X = np.asarray(values, dtype=float).reshape(1, -1)

    try:
        prediction = int(model.predict(X)[0])
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Mo hinh khong the xu ly du lieu dau vao.",
        )

    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = float(probabilities[class_index])
    else:
        confidence = None

    return {
        "prediction": prediction,
        "label": (
            "Malignant (nhan ac tinh trong du lieu)"
            if prediction == 0
            else "Benign (nhan lanh tinh trong du lieu)"
        ),
        "confidence": (
            round(confidence * 100, 2)
            if confidence is not None
            else None
        ),
        "probabilities": (
            {
                str(int(class_id)): round(float(probability) * 100, 2)
                for class_id, probability in zip(
                    model.classes_, probabilities
                )
            }
            if probabilities is not None
            else {}
        ),
        "disclaimer": (
            "Ket qua phan loai du lieu hoc tap, "
            "khong phai chan doan y khoa."
        ),
    }
