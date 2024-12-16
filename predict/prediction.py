import joblib
import numpy as np
from typing import Any
import os
import requests
import sys

# Ensure the directory containing rf_pipeline.py is in sys.path
module_dir = os.path.dirname(os.path.abspath(__file__))  # Current directory of prediction.py
sys.path.append(module_dir)
from rf_pipeline import RandomForestPipeline

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
            session = requests.Session()
            response = session.get(url, stream=True)

            # Handle Google Drive specific responses
            if "drive.google.com" in url:
                confirm_token = None
                for key, value in response.cookies.items():
                    if key.startswith("download_warning"):
                        confirm_token = value
                        break

                if confirm_token:
                    url = f"{url}&confirm={confirm_token}"
                    response = session.get(url, stream=True)

            response.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Model downloaded and saved to {local_path}")

        # Ensure the file is fully written and accessible
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Failed to download the model to {local_path}")

        # Load the model using joblib
        print(f"Loading model from {local_path}...")
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
