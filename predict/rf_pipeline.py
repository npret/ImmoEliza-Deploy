import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, Optional


class RandomForestPipeline:
    """
    Pipeline for preprocessing, training and evaluating the Random Forest model with polynomial features.
    """

    def __init__(
        self,
        degree: int = 2,
        max_depth: Optional[int] = 10,
        min_samples_leaf: int = 2,
        min_samples_split: int = 2,
        n_estimators: int = 300,
        random_state: int = 42,
    ) -> None:
        """
        Initialize random forest pipeline with polynomial features.

        Args:
            degree (int): Degree of polynomial features. Default = 2.
            max_depth (Optional[int]): Max depth of the forest. Default = 10.
            min_samples_leaf (int): Min samples required at leaf node. Default = 2.
            min_samples_split (int): Min samples required to split a node. Default = 2.
            n_estimators (int): Number of trees in the forest. Default = 300.
            randome_state (int): Random seed for reproducibilty. Default = 42.
        """

        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.model = RandomForestRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.feature_names: Optional[np.ndarray] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

    def preprocess(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply polynomial features transformation to the dataset

        Args:
            X (pd.DataFrame): Input features

        Returns:
            np.ndarray: Transformed polynomial features.
        """

        X_poly = self.poly.fit_transform(X)
        self.feature_names = self.poly.get_feature_names_out(input_features=X.columns)
        return X_poly

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> None:
        """
        Split data, train model and store splits.

            Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable (Price)
            test_size (float): Proportion of data used as the test set. Default = 0.2
        """
        X_poly = self.preprocess(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_poly, y, test_size=test_size, random_state=42
        )
        # Apply log transformation to the target variable
        self.y_train = np.log1p(self.y_train)
        self.y_test = np.log1p(self.y_test)

        # Define a high-price threshold (top 10% of prices)
        high_price_threshold = self.y_train.quantile(0.90)

        # Compute sample weights
        sample_weights = self.y_train.apply(
            lambda price: 2 if price > high_price_threshold else 1
        )

        # Train the model with weighted loss
        self.model.fit(self.X_train, self.y_train, sample_weight=sample_weights)

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model and return RMSE, MAE, R², MAPE & sMAPE scores.

            Returns:
            Dict[str, Dict[str, float]]: A dictionary containing metrics for training and test sets.
        """

        def mape(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        def smape(y_true, y_pred):
            return 100 * np.mean(
                2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
            )

        # Predict on training and test set
        y_train_pred_log = self.model.predict(self.X_train)
        y_test_pred_log = self.model.predict(self.X_test)

        # Reverse log transformations safely
        y_train_pred = np.expm1(np.clip(y_train_pred_log, a_min=None, a_max=20))
        y_test_pred = np.expm1(np.clip(y_test_pred_log, a_min=None, a_max=20))
        y_train_actual = np.expm1(np.clip(self.y_train, a_min=None, a_max=20))
        y_test_actual = np.expm1(np.clip(self.y_test, a_min=None, a_max=20))

        metrics = {
            "training": {
                "MAE": mean_absolute_error(y_train_actual, y_train_pred),
                "RMSE": np.sqrt(mean_squared_error(y_train_actual, y_train_pred)),
                "R²": r2_score(y_train_actual, y_train_pred),
                "MAPE": mape(y_train_actual, y_train_pred),
                "sMAPE": smape(y_train_actual, y_train_pred),
            },
            "test": {
                "MAE": mean_absolute_error(y_test_actual, y_test_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test_actual, y_test_pred)),
                "R²": r2_score(y_test_actual, y_test_pred),
                "MAPE": mape(y_test_actual, y_test_pred),
                "sMAPE": smape(y_test_actual, y_test_pred),
            },
        }

        return metrics

    def feature_importance(self) -> pd.DataFrame:
        """
        Return Feature Importance scores.

            Returns:
                pd.DataFrame: Dataframe containing features and their importance scores.
        """
        importances = self.model.feature_importances_
        feature_importances = pd.DataFrame(
            {"Feature": self.feature_names, "Importance": importances * 100}
        )
        return feature_importances.sort_values(by="Importance", ascending=False)

    def save_predictions(self, file_name: str = "model_predictions.csv") -> None:
        """
        Save predicted values compared to actual values + their difference to a .csv file.

            Args:
            file_name (str): Name of the file where predictions are stored. Default = "model_predictions.csv".
        """
        # Predict on test set (log-transformed scale)
        y_pred_log = self.model.predict(self.X_test)
        y_pred = np.expm1(
            np.clip(y_pred_log, a_min=None, a_max=20)
        )  # Reverse log transformation safely
        y_actual = np.expm1(np.clip(self.y_test, a_min=None, a_max=20))

        comparison = pd.DataFrame(
            {
                "Actual Price": np.expm1(np.clip(self.y_test, a_min=None, a_max=20)),
                "Predicted Price": y_pred,
                "Difference": abs(
                    np.expm1(np.clip(self.y_test, a_min=None, a_max=20)) - y_pred
                ),
            }
        )
        comparison.to_csv(file_name, index=False)
        print(f"Predictions saved to {file_name}")
