import pandas as pd
import numpy as np
from typing import Any, Dict


class DataPreprocessor:
    """
    Handles preprocessing of property data.

    Attributes:
        type_mapping: Maps property types to numerical values.
        state_mapping: Maps property states to numerical values.
        region_mapping: Maps regions to numerical values.
        municipality_mapping: Maps municipalities to regions and codes.
        municipality_income_mapping: Maps municipalities to average income values.
    """

    def __init__(self):
        """
        Initializes the DataPreprocessor with mappings for categorical features.
        """
        self.type_mapping = {"Apartment": 0, "House": 1}
        self.state_mapping = {
            "Good": 1,
            "Unknown": 2,
            "As new": 3,
            "To renovate": 4,
            "To be done up": 5,
            "Just renovated": 6,
            "To restore": 7,
        }
        self.region_mapping = {"Brussel": 0, "Flanders": 1, "Wallonia": 2}
        self.municipality_mapping = {
            "Antwerpen": {"region": "Flanders", "code": 0},
            "Brussel": {"region": "Brussel", "code": 1},
            "Henegouwen": {"region": "Wallonia", "code": 2},
            "Limburg": {"region": "Flanders", "code": 3},
            "Luik": {"region": "Wallonia", "code": 4},
            "Luxemburg": {"region": "Wallonia", "code": 5},
            "Namen": {"region": "Wallonia", "code": 6},
            "Oost-Vlaanderen": {"region": "Flanders", "code": 7},
            "Vlaams-Brabant": {"region": "Flanders", "code": 8},
            "Waals-Brabant": {"region": "Wallonia", "code": 9},
            "West-Vlaanderen": {"region": "Flanders", "code": 10},
        }
        self.municipality_income_mapping = {
            "Antwerpen": 31370.66,
            "Brussel": 29213.63,
            "Henegouwen": 25779.83,
            "Limburg": 31620.44,
            "Luik": 29132.64,
            "Luxemburg": 32628.53,
            "Namen": 27685.44,
            "Oost-Vlaanderen": 30710.00,
            "Vlaams-Brabant": 36105.99,
            "Waals-Brabant": 39882.77,
            "West-Vlaanderen": 30269.35,
        }

    def preprocess(self, data: dict) -> pd.DataFrame:
        """
        Preprocesses input data for model prediction.

        Args:
            data (Dict[str, Any]): Raw input data from the user.

        Returns:
            pd.DataFrame: Preprocessed data ready for the model's pipeline.
        """
        # Map categorical inputs to numerical values
        property_type = self.type_mapping[data["property_type"]]
        bedrooms = data["bedrooms"]
        kitchen_equipped = 1 if data["kitchen_equipped"] else 0
        state = self.state_mapping[data["state"]]
        facades = data["facades"]
        swimming_pool = 1 if data["swimming_pool"] else 0
        region = self.region_mapping[data["region"]]
        municipality = data["municipality"]
        municipality_code = self.municipality_mapping[municipality]["code"]
        avg_income = self.municipality_income_mapping.get(municipality, 0)
        bedroom_bin_code = 1 if bedrooms <= 2 else 2 if bedrooms <= 4 else 3
        log_living_area = np.log(data["living_area"])
        sqrt_total_outdoor_area = data["Sqrt_Total_Outdoor_Area"]

        # Create the feature DataFrame
        features = pd.DataFrame(
            {
                "Type": [property_type],
                "Bedrooms": [bedrooms],
                "Is_Equiped_Kitchen": [kitchen_equipped],
                "State": [state],
                "Facades": [facades],
                "Swim_pool": [swimming_pool],
                "Municipality": [municipality_code],
                "Region": [region],
                "Average_Income": [avg_income],
                "Bedroom_Bin_Code": [bedroom_bin_code],
                "Log_Living_Area": [log_living_area],
                "Sqrt_Total_Outdoor_Area": [sqrt_total_outdoor_area],
            }
        )

        return features
