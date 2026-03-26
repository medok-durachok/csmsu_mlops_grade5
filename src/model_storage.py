import pickle
import json
from pathlib import Path
from datetime import datetime


class ModelStorage:
    def __init__(self, storage_dir="models"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "models_metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"models": {}, "current_version": None}

    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save_model(self, model, model_name, metrics, preprocessor=None, version_notes=""):
        version_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_dir = self.storage_dir / version_id
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        if preprocessor:
            preprocessor_path = model_dir / "preprocessor.pkl"
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)

        model_metadata = {
            "model_name": model_name,
            "version_id": version_id,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "preprocessor_saved": preprocessor is not None,
            "version_notes": version_notes
        }

        self.metadata["models"][version_id] = model_metadata
        self.metadata["current_version"] = version_id
        self._save_metadata()

        return version_id

    def load_model(self, version_id=None):
        if version_id is None:
            version_id = self.metadata["current_version"]

        if version_id not in self.metadata["models"]:
            raise ValueError(f"Model version {version_id} not found")

        model_dir = self.storage_dir / version_id
        model_path = model_dir / "model.pkl"

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        preprocessor = None
        preprocessor_path = model_dir / "preprocessor.pkl"
        if preprocessor_path.exists():
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)

        return model, preprocessor, self.metadata["models"][version_id]

    def get_model_versions(self):
        return list(self.metadata["models"].keys())

    def get_model_info(self, version_id):
        return self.metadata["models"].get(version_id)

    def compare_models(self, version1, version2):
        info1 = self.get_model_info(version1)
        info2 = self.get_model_info(version2)

        if not info1 or not info2:
            raise ValueError("One or both model versions not found")

        comparison = {
            "version1": info1,
            "version2": info2,
            "metrics_comparison": {}
        }

        for metric in info1["metrics"]:
            if metric in info2["metrics"]:
                val1 = info1["metrics"][metric]
                val2 = info2["metrics"][metric]
                comparison["metrics_comparison"][metric] = {
                    "version1": val1,
                    "version2": val2,
                    "difference": val2 - val1,
                    "improvement": val2 > val1 if metric in ["r2"] else val2 < val1
                }

        return comparison


class QualityControl:
    def __init__(self, model_storage):
        self.model_storage = model_storage

    def check_model_quality(self, version_id, quality_thresholds=None):
        if quality_thresholds is None:
            quality_thresholds = {
                "r2": 0.6,
                "rmse": 1000,
                "mae": 500
            }

        model_info = self.model_storage.get_model_info(version_id)
        if not model_info:
            raise ValueError(f"Model version {version_id} not found")

        metrics = model_info["metrics"]
        quality_report = {
            "version_id": version_id,
            "model_name": model_info["model_name"],
            "quality_checks": {},
            "overall_quality": "good",
            "failed_checks": []
        }

        for metric, threshold in quality_thresholds.items():
            if metric in metrics:
                value = metrics[metric]
                if metric in ["r2"]:
                    passed = value >= threshold
                else:
                    passed = value <= threshold

                quality_report["quality_checks"][metric] = {
                    "value": value,
                    "threshold": threshold,
                    "passed": passed
                }

                if not passed:
                    quality_report["failed_checks"].append(metric)
                    quality_report["overall_quality"] = "poor"

        return quality_report

    def monitor_model_drift(self, current_version_id, reference_version_id=None):
        if reference_version_id is None:
            versions = self.model_storage.get_model_versions()
            if len(versions) < 2:
                return {"status": "insufficient_data", "message": "Need at least 2 model versions"}
            reference_version_id = versions[-2]

        comparison = self.model_storage.compare_models(reference_version_id, current_version_id)

        drift_report = {
            "current_version": current_version_id,
            "reference_version": reference_version_id,
            "drift_detected": False,
            "significant_changes": []
        }

        for metric, comp in comparison["metrics_comparison"].items():
            change_percent = abs(comp["difference"] / comp["version1"]) * 100 if comp["version1"] != 0 else 100

            if change_percent > 10:
                drift_report["significant_changes"].append({
                    "metric": metric,
                    "change_percent": change_percent,
                    "improvement": comp["improvement"]
                })
                drift_report["drift_detected"] = True

        return drift_report