import joblib
import numpy as np
from typing import Any, Dict


class PricePredictor:
    """
    Handles price prediction using a pre-trained model.

    Attributes:
        pipeline: The loaded pipeline containing preprocessing and model steps.
        model: The trained model extracted from the pipeline.
    """

    def __init__(self, model_path: str = "model/trained_model.pkl"):
        """
        Initializes the PricePredictor by loading the pipeline and model.

        Args:
            model_path (str): Path to the trained model file.
        """
        pipeline = joblib.load(model_path)
        self.pipeline = pipeline

        # Extract the model from the pipeline
        if hasattr(pipeline, "model"):
            self.model = pipeline.model
        elif hasattr(pipeline, "steps"):
            self.model = pipeline.steps[-1][1]
        else:
            # If no known model attribute, assume pipeline itself is the model
            self.model = pipeline

    def predict(self, features, preprocessor):
        """
        Predicts the price of a property.

        Args:
            features (Dict[str, Any]): The raw input features from the user.
            preprocessor (Any): The DataPreprocessor object to preprocess features.

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
