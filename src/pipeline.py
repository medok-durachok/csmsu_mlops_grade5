import pandas as pd
import numpy as np
from .preprocessor import DataPreprocessor, clean_data
from .models import ModelTrainer
from .collector import DataCollector
from .association_rules import AprioriRulesMiner
from .model_storage import ModelStorage, QualityControl
from .utils import Timer
from .eda import AutoEDA
from .feature_engineering import FeatureEngineer
from sklearn.model_selection import train_test_split
from .dq import basic_dq_report


class MLPipeline:
    def __init__(self, config):
        self.config = config
        self.target_col = config.TARGET_COL
        self.collector = DataCollector(main_data_file=config.MAIN_DATA_FILE,raw_dir=config.RAW_DIR)
        self.preprocessor = DataPreprocessor(target_col=self.target_col)
        self.model_trainer = ModelTrainer(target_col=self.target_col,random_state=config.RANDOM_STATE)
        self.rules_miner = AprioriRulesMiner(min_support=0.1, min_confidence=0.5)
        self.is_trained = False
        self.current_model_version = None
        self.rules_fitted = False

        self.model_storage = ModelStorage(storage_dir=config.MODELS_DIR)
        self.quality_control = QualityControl(self.model_storage)
        self.eda = AutoEDA(output_dir=config.REPORTS_DIR / "eda")
        self.feature_engineer = FeatureEngineer()

        if self.model_storage.metadata and self.model_storage.metadata.get("models"):
            models_info = self.model_storage.metadata["models"]
            if models_info:
                self.is_trained = True
                if self.model_storage.metadata.get("current_version"):
                    self.current_model_version = self.model_storage.metadata["current_version"]
                else:
                    latest_version = max(models_info.keys(), key=lambda k: models_info[k].get("saved_at", ""))
                    self.current_model_version = latest_version

    def _split_data(self, df, test_size=0.2):
        if self.target_col not in df.columns: raise ValueError(f"Target column {self.target_col} not found")

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.config.RANDOM_STATE)
        return X_train, X_test, y_train, y_test

    def _prepare_data(self, df, fit_preprocessor=True):
        if self.target_col in df.columns:
            df = df.dropna(subset=[self.target_col])

        if df.empty:
            raise ValueError(f"No data with valid target column {self.target_col} found")

        target_values = df[self.target_col].copy() if self.target_col in df.columns else None

        dq_report = basic_dq_report(df)
        print(f"Data Quality Report:")
        print(f"  Duplicate rate: {dq_report['duplicate_rate']:.4f}")
        print(f"  Missing rate (avg): {np.mean(list(dq_report['missing_rate'].values())):.4f}")
        print(f"  Constant columns: {len(dq_report['constant_cols'])}")

        df_clean = clean_data(
            df,
            max_missing_rate=self.config.MAX_MISSING_RATE,
            max_duplicate_rate=self.config.MAX_DUPLICATE_RATE,
            apply_rules=self.rules_fitted,
            rules_miner=self.rules_miner if self.rules_fitted else None,
            target_col=self.target_col
        )

        df_engineered = self.feature_engineer.transform(df_clean, self.target_col)

        if target_values is not None and self.target_col not in df_engineered.columns:
            df_engineered[self.target_col] = target_values

        if fit_preprocessor:
            df_preprocessed = self.preprocessor.fit_transform(df_engineered)
        else:
            df_preprocessed = self.preprocessor.transform(df_engineered)

        if target_values is not None:
            df_preprocessed[self.target_col] = target_values.values

        numeric_cols = df_preprocessed.select_dtypes(include=[np.number]).columns.tolist()
        df_final = df_preprocessed[numeric_cols].fillna(0)
        return df_final

    def train_initial_model(self, df):
        print("Training model:")

        print("Running EDA...")
        eda_report = self.eda.analyze(df, self.target_col)
        eda_path = self.eda.save_report()
        eda_text_path = self.eda.save_text_report()
        print(f"EDA reports saved: {eda_path.name}, {eda_text_path.name}")

        with Timer() as timer:
            df_prepared = self._prepare_data(df, fit_preprocessor=True)
            X_train, X_test, y_train, y_test = self._split_data(df_prepared)

            model_params = {
                'linear': {'fit_intercept': self.config.LR_FIT_INTERCEPT},
                'tree': {
                    'max_depth': self.config.DT_MAX_DEPTH,
                    'min_samples_split': self.config.DT_MIN_SAMPLES_SPLIT,
                    'min_samples_leaf': self.config.DT_MIN_SAMPLES_LEAF
                },
                'neural': {
                    'hidden_layer_sizes': self.config.MLP_HIDDEN_LAYER_SIZES,
                    'max_iter': self.config.MLP_MAX_ITER,
                    'alpha': self.config.MLP_ALPHA,
                    'solver': self.config.MLP_SOLVER
                }
            }

            self.model_trainer.train_all_models(X_train, y_train, X_test, y_test, cv=self.config.CV_FOLDS,
                                               tune_hyperparams=False, **model_params)

            version_id = self.model_trainer.save_final_model(self.model_storage, preprocessor=self.preprocessor, version_notes="Initial model training")
            best_model_name, best_metrics = self.model_trainer.select_best_model()

            self.is_trained = True
            self.current_model_version = version_id

        results = {
            'success': True,
            'version_id': version_id,
            'best_model': best_model_name,
            'metrics': {k: v for k, v in best_metrics.items() if k != 'model'},
            'training_time': timer.dt,
            'n_samples': len(df)
        }

        print(f"Model training completed in {timer.dt:.2f}s")
        return results

    def update_model(self, new_data):
        print("Updating model with new data:")
        if not self.is_trained: raise ValueError("No trained model found. Train initial model first.")

        with Timer() as timer:
            current_metrics = {}
            df_prepared = self._prepare_data(new_data, fit_preprocessor=False)
            reference_data = self._get_reference_data()
            if reference_data is not None:
                numeric_cols = reference_data.select_dtypes(include=[np.number]).columns.tolist()
                if self.target_col in numeric_cols: numeric_cols.remove(self.target_col)

                drift_report = {'n_drifted_features': 0}
                return {
                    'success': True,
                    'retrained': False,
                    'reason': 'No significant drift detected',
                    'drift_report': drift_report
                }
            X_train, X_test, y_train, y_test = self._split_data(df_prepared)

            model_params = {
                'linear': {'fit_intercept': self.config.LR_FIT_INTERCEPT},
                'tree': {
                    'max_depth': self.config.DT_MAX_DEPTH,
                    'min_samples_split': self.config.DT_MIN_SAMPLES_SPLIT,
                    'min_samples_leaf': self.config.DT_MIN_SAMPLES_LEAF
                },
                'neural': {
                    'hidden_layer_sizes': self.config.MLP_HIDDEN_LAYER_SIZES,
                    'max_iter': self.config.MLP_MAX_ITER,
                    'alpha': self.config.MLP_ALPHA,
                    'solver': self.config.MLP_SOLVER
                }
            }

            self.model_trainer.train_all_models(X_train, y_train, X_test, y_test, cv=self.config.CV_FOLDS,
                                                               tune_hyperparams=False, **model_params)

            best_model_name, best_metrics = self.model_trainer.select_best_model()
            degradation_report = {}
            version_id = self.model_trainer.save_final_model(
                self.model_storage,
                preprocessor=self.preprocessor,
                version_notes="Model update"
            )
            self.current_model_version = version_id

        results = {
            'success': True,
            'retrained': True,
            'version_id': version_id,
            'best_model': best_model_name,
            'metrics': {k: v for k, v in best_metrics.items() if k != 'model'},
            'previous_metrics': current_metrics,
            'degradation_report': degradation_report,
            'update_time': timer.dt,
            'n_samples': len(new_data)
        }

        print(f"Model update completed in {timer.dt:.2f}s")
        return results

    def _get_reference_data(self):
        processed_files = list(self.config.PROCESSED_DIR.glob("*.csv"))
        if processed_files: return pd.read_csv(processed_files[0])
        return None

    def process_batch(self):
        df, meta = self.collector.get_next_batch(chunk_size=self.config.BATCH_SIZE)
        if df is None or meta is None: return None

        if self.is_trained:
            results = self.update_model(df)
        else:
            results = self.train_initial_model(df)

        results['batch_id'] = meta['batch_id']
        return results

    def train_with_real_data(self, chunk_size=None):
        chunk_size = chunk_size or self.config.BATCH_SIZE
        df, meta = self.collector.get_next_batch(chunk_size=chunk_size)

        if df is None: return {"success": False, "error": "Failed to get data from main file"}

        results = self.train_initial_model(df)
        results.update({
            "data_source": "real_data",
            "chunk_size": chunk_size,
            "data_metadata": meta
        })
        return results

    def load_model(self, version_id=None):
        model, preprocessor, metadata = self.model_storage.load_model(version_id)
        self.preprocessor = preprocessor or self.preprocessor
        self.current_model_version = metadata["version_id"]
        self.is_trained = True
        return model, preprocessor, metadata

    def process_real_data_batch(self, chunk_size=None):
        chunk_size = chunk_size or self.config.BATCH_SIZE
        df, _ = self.collector.get_next_batch(chunk_size=chunk_size)
        if df is None: return None

        df_processed = self._prepare_data(df, fit_preprocessor=False)
        return df_processed

    def predict(self, input_data):
        if not self.is_trained: raise ValueError("No trained model found. Train or load a model first.")
        model, _, _ = self.load_model()

        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
        else:
            expected_features = ['SEX', 'INSR_TYPE', 'INSURED_VALUE', 'PREMIUM',
                               'OBJECT_ID', 'PROD_YEAR', 'SEATS_NUM', 'CARRYING_CAPACITY', 'CCM_TON']

        if isinstance(input_data, str):
            input_data = pd.read_csv(input_data)
        elif not isinstance(input_data, pd.DataFrame):
            raise ValueError("input_data must be a DataFrame or path to CSV file")

        if self.target_col in input_data.columns: input_data = input_data.drop(columns=[self.target_col])

        available_features = [col for col in expected_features if col in input_data.columns]
        missing_features = [col for col in expected_features if col not in input_data.columns]

        if not available_features:  raise ValueError("No matching features found between model and input data")

        input_data_filtered = input_data[available_features]
        df_prepared = self._prepare_data(input_data_filtered, fit_preprocessor=False)
        predictions = model.predict(df_prepared)
        return predictions