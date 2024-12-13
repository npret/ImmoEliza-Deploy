# Challenge: API Deployment

## Repository: `challenge-app-deployment`

---

## Project Description

The real estate company **ImmoEliza** is excited about the regression model developed to predict property prices. This project focuses on deploying that model via a user-friendly Streamlit app, enabling their team to simulate property valuations interactively.

The app will:
- Allow manual feature entry to predict property prices.
- Simulate potential property valuations to support market analysis.
- Deploy locally or via Streamlit Community Cloud.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/npret/ImmoEliza-Deploy.git
   cd ImmoEliza-Deploy
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   streamlit --version
   ```

---

## Usage

1. **Run the Application**
   ```bash
   streamlit run app.py
   ```

2. **Access the App**
   - Open the URL provided in the terminal (e.g., `http://localhost:8501`) in your browser.

3. **Features of the App**
   - Enter property details interactively.
   - View the predicted property price in real-time.
   - Analyze results with a clean and intuitive UI.

---

## Visuals

Here is a glimpse of the app:

![App Screenshot](path/to/screenshot.png)

---

## Contributors

- **Nicole Pretorius**
  - Role: Developer
  - Responsibilities:
    - Designed and implemented the Streamlit app interface.
    - Integrated the machine learning model for property price predictions.
    - Developed reusable preprocessing and prediction modules.
    - Ensured code adheres to professional standards (OOP, typing, docstrings).
    - Deployed the application locally and prepared for potential cloud deployment.

---

## Timeline

- **Project Start**: 11/12/2024
- **Deadline**: 13/12/2024

---

## Personal Situation

This project was tackled solo and required integrating a pre-trained regression model into a deployable API via Streamlit. Key learnings include:
- Creating reusable preprocessing and prediction modules.
- Designing a user-friendly interface.
- Debugging and optimizing deployment strategies for large models.

---

### Deliverables Checklist

- [x] **Streamlit App**: Deployed and functional.
- [x] **README**: Fully detailed and professional.
- [x] **Documentation**: Clear and comprehensive.

