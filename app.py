import streamlit as st
import numpy as np
from preprocessing.cleaning_data import DataPreprocessor
from predict.prediction import PricePredictor
from streamlit_toggle import st_toggle_switch
import locale

# Configure locale for price formatting
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

# Define feature icons
feature_icons = {
    "Property Type": "🏠",
    "Bedrooms": "🛏️",
    "Kitchen Equipped": "🍳",
    "State": "🏚️",
    "Facades": "🚪",
    "Swimming Pool": "🏊",
    "Municipality": "📍",
    "Region": "🌍",
    "Living Area": "📐",
    "Total Outdoor Area": "🌳",
}


class PropertyApp:
    """
    Streamlit App for Property Price Prediction.
    """

    def __init__(self, model_path: str):
        self.preprocessor = DataPreprocessor()
        self.predictor = PricePredictor(model_path)

    @staticmethod
    def format_price(price: float) -> str:
        """
        Format price with a space as the thousand separator.
        Args:
            price (float): The price to format.
        Returns:
            str: The formatted price string.
        """
        return (
            f"€{locale.format_string('%.2f', price, grouping=True).replace(',', ' ')}"
        )

    @staticmethod
    def get_size_category(area: int) -> str:
        """
        Categorizes a property size based on living area.
        Args:
            area (int): The living area in square meters.
        Returns:
            str: The size category description.
        """
        if area <= 20:
            return "Tiny Apartment"
        elif area <= 50:
            return "Small Apartment"
        elif area <= 100:
            return "Medium Apartment"
        elif area <= 300:
            return "Regular House"
        elif area <= 500:
            return "Large House"
        elif area <= 1000:
            return "Villa"
        else:
            return "Mansion"

    def input_features(self) -> dict:
        """
        Collects user input features via Streamlit's sidebar.
        Returns:
            dict: The collected input features.
        """
        st.sidebar.header("Property Features")

        property_type = st.sidebar.selectbox(
            "House or Apartment?", ["Apartment", "House"]
        )
        bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=0, value=2)

        # Toggle switches for binary features
        with st.sidebar:
            kitchen_equipped = st_toggle_switch(
                label="Is the kitchen equipped?",
                key="kitchen_toggle",
                default_value=False,
                label_after=True,
            )
            swimming_pool = st_toggle_switch(
                label="Is there a swimming pool?",
                key="swimming_pool_toggle",
                default_value=False,
                label_after=True,
            )

        state = st.sidebar.selectbox(
            "Condition of the building",
            [
                "Good",
                "Unknown",
                "As new",
                "To renovate",
                "To be done up",
                "Just renovated",
                "To restore",
            ],
        )
        facades = st.sidebar.number_input("Number of Facades", min_value=1, value=2)
        region = st.sidebar.selectbox(
            "Region", list(self.preprocessor.region_mapping.keys())
        )

        # Dynamically filter municipalities based on the selected region
        municipalities = [
            m
            for m, details in self.preprocessor.municipality_mapping.items()
            if details["region"] == region
        ]
        municipality = st.sidebar.selectbox("Municipality", municipalities)

        # Living Area slider with size category
        living_area = st.sidebar.slider("Living Area (sq. meters)", 10, 2000, 50)
        size_category = self.get_size_category(living_area)
        st.sidebar.write(f"🏠 This is a **{size_category}**.")

        terrace_area = st.sidebar.slider("Terrace Area (sq. meters)", 0, 2000, 0)
        garden_area = st.sidebar.slider("Garden Area (sq. meters)", 0, 1000, 0)

        # Calculate total outdoor area and its square root transformation
        total_outdoor_area = terrace_area + garden_area
        sqrt_total_outdoor_area = np.sqrt(total_outdoor_area)

        return {
            "property_type": property_type,
            "bedrooms": bedrooms,
            "kitchen_equipped": kitchen_equipped,
            "state": state,
            "facades": facades,
            "swimming_pool": swimming_pool,
            "region": region,
            "municipality": municipality,
            "living_area": living_area,
            "Total_Outdoor_Area": int(total_outdoor_area),
            "Sqrt_Total_Outdoor_Area": sqrt_total_outdoor_area,
        }

    def display_selected_features(self, features: dict) -> None:
        """
        Displays the selected features with appropriate formatting and icons.
        Args:
            features (dict): The user-selected features.
        """
        st.subheader("Selected Features:")
        cols = st.columns(2)

        for i, (key, value) in enumerate(features.items()):
            if key == "Sqrt_Total_Outdoor_Area":  # Skip transformed feature for display
                continue

            formatted_key = key.replace("_", " ").title()
            icon = feature_icons.get(formatted_key, "")
            if isinstance(value, bool):  # Convert binary features to "Yes"/"No"
                value = "Yes" if value else "No"
            if key in [
                "living_area",
                "Total_Outdoor_Area",
            ]:  # Append m² for area features
                value = f"{value} m²"

            with cols[i % 2]:
                st.write(f"{icon} **{formatted_key}:** {value}")

    def run(self):
        """
        Runs the Streamlit app.
        """
        st.image("images/logo.png", width=300)
        st.title("Property Price Predictor")

        # Collect user input
        features = self.input_features()

        # Display user selections
        self.display_selected_features(features)

        # Predict price
        if st.button("Predict Price"):
            try:
                predicted_price = self.predictor.predict(features, self.preprocessor)
                formatted_price = self.format_price(predicted_price)
                st.success(f"Predicted Price: {formatted_price}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")


# Run the app
if __name__ == "__main__":
    app = PropertyApp("model/trained_model.pkl")
    app.run()
