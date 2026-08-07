from functools import lru_cache
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

MP_MODEL_DIR = BASE_DIR / "mp_model"
NP_MODEL_DIR = BASE_DIR / "np_model"

MODEL_FILES = {
    "filtered_mp_dev_v1": MP_MODEL_DIR / "filtered_rf_v1.joblib",
    "salt_mp_v1": MP_MODEL_DIR / "salt_rf_v1.joblib",
    "np_v1": NP_MODEL_DIR / "np_rf_v1.joblib",
}


def model_path(model_name):
    if model_name not in MODEL_FILES:
        raise KeyError(f"Unknown model: {model_name}")
    return MODEL_FILES[model_name]


@lru_cache(maxsize=8)
def load_bundle(model_name):
    path = model_path(model_name)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file is not deployed: {path}. "
            f"Upload the expected model file to the correct folder."
        )

    obj = joblib.load(path)

    if isinstance(obj, dict) and "model" in obj:
        return obj

    return {
        "model": obj,
        "model_name": model_name,
        "model_version": model_name,
        "feature_names": None,
        "class_labels": None,
    }


def available_models():
    return {
        name: {
            "filename": path.name,
            "folder": path.parent.name,
            "present": path.exists(),
        }
        for name, path in MODEL_FILES.items()
    }
