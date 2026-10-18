
from pathlib import Path
import json

import joblib
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.metrics import make_scorer

# 1. Xác định thư mục dự án
BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "artifacts"
REPORT_DIR = BASE_DIR / "reports"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Nạp bộ dữ liệu Breast Cancer Wisconsin
dataset = load_breast_cancer(as_frame=True)

X = dataset.data
y = dataset.target

FEATURE_NAMES = list(X.columns)

print("=" * 60)
print("ONCOSENSE AI - HUAN LUYEN MO HINH SVM")
print("=" * 60)
print("Kich thuoc du lieu:", X.shape)
print("So dac trung:", len(FEATURE_NAMES))
print("So mau moi lop:")
print(y.value_counts().sort_index())
print("Nhan:", dataset.target_names)
print("Gia tri thieu:", int(X.isna().sum().sum()))

# 3. Chia du lieu: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# 4. Tao pipeline: chuan hoa + SVM
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    ))
])

# 5. Tim bo tham so phu hop bang cross-validation
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.001, 0.01, 0.1]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Uu tien recall cua lop 0 = malignant (ac tinh)
malignant_recall = make_scorer(
    recall_score,
    pos_label=0
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=malignant_recall,
    cv=cv,
    n_jobs=-1,
    refit=True
)

print("\nDang tim tham so tot nhat...")
search.fit(X_train, y_train)

model = search.best_estimator_

print("Tham so tot nhat:", search.best_params_)
print("Recall ac tinh trung binh CV:", round(search.best_score_, 4))

# 6. Du doan tren tap test
y_pred = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# SVC classes thuong la [0, 1]; tim cot theo nhan thay vi doan cot
classes = list(model.named_steps["svc"].classes_)
malignant_col = classes.index(0)
benign_col = classes.index(1)

prob_malignant = probabilities[:, malignant_col]
prob_benign = probabilities[:, benign_col]

# 7. Tinh cac chi so danh gia
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred, pos_label=0)
specificity = recall_score(y_test, y_pred, pos_label=1)
precision_malignant = precision_score(
    y_test, y_pred, pos_label=0, zero_division=0
)
f1_malignant = f1_score(
    y_test, y_pred, pos_label=0, zero_division=0
)
roc_auc = roc_auc_score(
    (y_test == 0).astype(int),
    prob_malignant
)

metrics = {
    "model_name": "OncoSense AI - SVM RBF",
    "model_version": "1.0.0",
    "dataset": "Breast Cancer Wisconsin Diagnostic",
    "total_samples": int(len(X)),
    "number_of_features": int(X.shape[1]),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "best_params": search.best_params_,
    "cv_malignant_recall": float(search.best_score_),
    "accuracy": float(accuracy),
    "sensitivity_malignant": float(sensitivity),
    "specificity_benign": float(specificity),
    "precision_malignant": float(precision_malignant),
    "f1_malignant": float(f1_malignant),
    "roc_auc_malignant": float(roc_auc),
    "confusion_matrix_labels": ["malignant", "benign"],
    "confusion_matrix": cm.tolist(),
    "sklearn_version": sklearn.__version__,
    "warning": (
        "Educational demonstration only. "
        "Not a medical diagnosis."
    )
}

print("\n" + "=" * 60)
print("KET QUA DANH GIA TREN TAP TEST")
print("=" * 60)
print(f"Accuracy:              {accuracy:.4f}")
print(f"Sensitivity malignant: {sensitivity:.4f}")
print(f"Specificity benign:    {specificity:.4f}")
print(f"Precision malignant:   {precision_malignant:.4f}")
print(f"F1 malignant:          {f1_malignant:.4f}")
print(f"ROC-AUC malignant:     {roc_auc:.4f}")
print("\nConfusion matrix (malignant, benign):")
print(cm)

# 8. Luu mo hinh va metadata
joblib.dump(model, ARTIFACT_DIR / "breast_cancer_svm.joblib")

metadata = {
    "model_name": metrics["model_name"],
    "model_version": metrics["model_version"],
    "feature_names": FEATURE_NAMES,
    "class_mapping": {
        "0": "malignant",
        "1": "benign"
    },
    "best_params": search.best_params_,
    "sklearn_version": sklearn.__version__,
    "warning": metrics["warning"]
}

with open(
    ARTIFACT_DIR / "metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

with open(
    REPORT_DIR / "model_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# 9. Luu bieu do confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Malignant", "Benign"]
)
disp.plot()
plt.title("OncoSense AI - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "confusion_matrix.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close()

# 10. Luu bieu do ROC
RocCurveDisplay.from_predictions(
    (y_test == 0).astype(int),
    prob_malignant,
    name="SVM - Malignant"
)
plt.title("OncoSense AI - ROC Curve")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "roc_curve.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close()

print("\nDa luu mo hinh:", ARTIFACT_DIR / "breast_cancer_svm.joblib")
print("Da luu metadata:", ARTIFACT_DIR / "metadata.json")
print("Da luu bao cao:", REPORT_DIR / "model_metrics.json")
print("Da luu bieu do:", REPORT_DIR)
print("\nHOAN TAT HUAN LUYEN!")
print("Luu y: day la mo hinh hoc tap, khong thay the bac si.")
