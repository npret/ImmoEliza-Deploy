import joblib
import numpy as np
from typing import Any
import os
import requests


class PricePredictor:
    """
    Handles price prediction using a pre-trained model.

    Attributes:
        pipeline: The loaded pipeline containing preprocessing and model steps.
        model: The trained model extracted from the pipeline.
    """

    def __init__(
        self, model_url: str, local_model_path: str = "model/trained_model.pkl"
    ):
        """
        Initializes the PricePredictor by downloading the model if not present locally.

        Args:
            model_url (str): URL to download the model from.
            local_model_path (str): Local path to store the downloaded model.
        """
        self.pipeline = self.load_model(model_url, local_model_path)

        # Extract the model from the pipeline
        if hasattr(self.pipeline, "model"):
            self.model = self.pipeline.model
        elif hasattr(self.pipeline, "steps"):
            self.model = self.pipeline.steps[-1][1]
        else:
            self.model = self.pipeline  # Assume pipeline itself is the model

    @staticmethod
    def load_model(url: str, local_path: str) -> Any:
        """
        Downloads the model from the given URL if not already present locally.

        Args:
            url (str): URL to download the model.
            local_path (str): Path to save the model.

        Returns:
            The loaded model.
        """
        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the model if it does not exist locally
        if not os.path.exists(local_path):
            print(f"Downloading model from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error if the request fails
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Load the model using joblib
        return joblib.load(local_path)

    def predict(self, features: Any, preprocessor: Any) -> float:
        """
        Predicts the price of a property.

        Args:
            features: Raw input features for prediction.
            preprocessor: Preprocessor to prepare the input data.

        Returns:
            float: The predicted property price.
        """
        # Preprocess the input features
        raw_features = preprocessor.preprocess(features)
        processed_features = self.pipeline.preprocess(raw_features)

        # Predict using the pipeline's model
        predicted_price = self.model.predict(processed_features)

        # Reverse log transformation
        return np.expm1(predicted_price[0])
