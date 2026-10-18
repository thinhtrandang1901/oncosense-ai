
from pathlib import Path
import json
import math

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field

# Đường dẫn gốc của dự án
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "breast_cancer_svm.joblib"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"

app = FastAPI(
    title="OncoSense AI",
    description=(
        "Dự án học tập minh họa phân loại dữ liệu "
        "Breast Cancer Wisconsin bằng mô hình SVM."
    ),
    version="1.0.0",
)

# Đọc mô hình và thông tin dữ liệu
model = None
metadata = {}

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

# Lấy đúng tên 30 đặc trưng từ bộ dữ liệu
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
FEATURE_NAMES = list(dataset.feature_names)

SAMPLE_FEATURES = {
    name: float(value)
    for name, value in zip(
        FEATURE_NAMES,
        dataset.data[0]
    )
}

@app.get("/sample")
def get_sample():
    return {"features": SAMPLE_FEATURES}

# Đường dẫn giao diện
TEMPLATE_DIR = ROOT / "app" / "templates"
STATIC_DIR = ROOT / "app" / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


class PredictionRequest(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health():
    """Kiểm tra API và trạng thái mô hình."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.get("/metadata")
def get_metadata():
    """Trả thông tin mô hình và chỉ số đánh giá."""
    return {
        "project": "OncoSense AI",
        "model_name": "Support Vector Machine (SVM)",
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "metrics": metadata.get("metrics", {}),
        "disclaimer": (
            "Chỉ phục vụ học tập, không dùng để chẩn đoán y khoa."
        ),
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Mở trang chủ."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"project_name": "OncoSense AI"},
    )


@app.post("/predict")
def predict(payload: PredictionRequest):
    """Phân loại một bộ 30 đặc trưng đầu vào."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Chưa có mô hình. Hãy chạy "
                "python scripts/train.py trước."
            ),
        )

    features = payload.features

    missing = [name for name in FEATURE_NAMES if name not in features]
    extra = [name for name in features if name not in FEATURE_NAMES]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Danh sách đặc trưng chưa hợp lệ.",
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    values = [features[name] for name in FEATURE_NAMES]

    if not all(math.isfinite(value) for value in values):
        raise HTTPException(
            status_code=422,
            detail="Các giá trị đầu vào phải là số hữu hạn.",
        )

    X = np.asarray(values, dtype=float).reshape(1, -1)

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    # Trong sklearn breast_cancer: 0 = malignant, 1 = benign
    class_index = list(model.classes_).index(prediction)
    confidence = float(probabilities[class_index])

    return {
        "prediction": prediction,
        "label": (
            "Malignant (nhãn ác tính trong dữ liệu)"
            if prediction == 0
            else "Benign (nhãn lành tính trong dữ liệu)"
        ),
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            str(int(class_id)): round(float(probability) * 100, 2)
            for class_id, probability in zip(
                model.classes_, probabilities
            )
        },
        "disclaimer": (
            "Kết quả phân loại dữ liệu học tập, "
            "không phải chẩn đoán y khoa."
        ),
    }
