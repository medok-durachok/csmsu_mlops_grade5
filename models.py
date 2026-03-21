import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelTrainer:
    def __init__(self, target_col="CLAIM_PAID", random_state=42):
        self.target_col = target_col
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}

    def _get_linear_regression(self, fit_intercept=True):
        return LinearRegression(fit_intercept=fit_intercept)

    def _get_decision_tree(self, max_depth=10,
                              min_samples_split=2,
                              min_samples_leaf=1):
        return DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state
        )

    def _get_neural_network(self, hidden_layer_sizes = (100, 50),
                               max_iter=500,
                               alpha=0.0001,
                               solver="adam"):
        valid_solvers = ["lbfgs", "sgd", "adam"]
        if solver not in valid_solvers:
            solver = "adam"
        return MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            alpha=alpha,
            solver=solver,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )

    def train_model(self, model_name, X_train, y_train, **model_params):
        if model_name == 'linear':
            model = self._get_linear_regression(**model_params)
        elif model_name == 'tree':
            model = self._get_decision_tree(**model_params)
        elif model_name == 'neural':
            model = self._get_neural_network(**model_params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train, y_train)
        self.models[model_name] = model
        return model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def cross_validate_model(self, model, X, y, cv=5):
        rmse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

        return {
            'cv_rmse_mean': float(rmse_scores.mean()),
            'cv_rmse_std': float(rmse_scores.std()),
            'cv_r2_mean': float(r2_scores.mean()),
            'cv_r2_std': float(r2_scores.std())
        }

    def train_all_models(self, X_train, y_train, X_test, y_test, cv=5, **model_params):
        model_configs = {
            'linear': model_params.get('linear', {}),
            'tree': model_params.get('tree', {}),
            'neural': model_params.get('neural', {})
        }

        for model_name, params in model_configs.items():
            try:
                model = self.train_model(model_name, X_train, y_train, **params)
                test_metrics = self.evaluate_model(model, X_test, y_test)
                cv_metrics = self.cross_validate_model(model, X_train, y_train, cv=cv)

                self.model_scores[model_name] = {
                    **test_metrics,
                    **cv_metrics,
                    'model': model
                }

            except Exception as e:
                self.model_scores[model_name] = {'error': str(e)}

        return self.model_scores

    def predict(self, X, model_name):
        model = self.models.get(model_name)
        if model is None: raise ValueError(f"Model {model_name} not found. Train it first.")

        return model.predict(X)