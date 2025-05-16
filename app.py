import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: black;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0px;
        background-color: #e8f5e9;
    }
    .confidence-meter {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 10px 0px;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        background-color: #4CAF50;
        text-align: center;
        color: white;
        line-height: 20px;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/320px-RMS_Titanic_3.jpg", width=200)
with col2:
    st.title("🚢 Titanic Survival Predictor")
    st.markdown("Predict whether a Titanic passenger would survive based on their characteristics.")

st.header("Passenger Information")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        pclass = st.selectbox("Ticket Class", [1, 2, 3], help="1 = First, 2 = Second, 3 = Third")
        age = st.number_input("Age", min_value=0, max_value=100, value=25)
        sibsp = st.number_input("Number of Siblings/Spouse on Board", min_value=0, max_value=8, value=0)

    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"])
        parch = st.number_input("Number of Parents/Children on Board", min_value=0, max_value=6, value=0)
        fare = st.number_input("Ticket Fare (£)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

    with col3:
        embarked = st.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"],
                                help="C = Cherbourg, Q = Queenstown, S = Southampton")

    submitted = st.form_submit_button("Predict Survival")

if submitted:
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [1 if sex == "Female" else 0],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    embarked_map = {
        "Cherbourg": 0,
        "Queenstown": 1,
        "Southampton": 2
    }
    input_data["Embarked"] = input_data["Embarked"].map(embarked_map)

    input_data = input_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.header("Prediction Result")

    if prediction == 1:
        result_text = "SURVIVED 🎉"
        result_emoji = "😊"
        color = "green"
    else:
        result_text = "DID NOT SURVIVE 😢"
        result_emoji = "😞"
        color = "red"

    confidence = max(probabilities) * 100

    st.markdown(f"""
        <div class="prediction-box" style="border-left: 5px solid {color};">
            <h2 style="color: {color};">{result_text} {result_emoji}</h2>
            <p style="color: {color};">The model predicts this passenger <strong>{'survived' if prediction == 1 else 'did not survive'}</strong> with a confidence of:</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="confidence-meter">
            <div class="confidence-fill" style="width: {confidence}%;">{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("About the Model")
st.markdown("""
This machine learning model predicts the survival of Titanic passengers based on the following features:

- **Ticket Class (Pclass)**: 1 = First, 2 = Second, 3 = Third
- **Gender (Sex)**: Male or Female
- **Age**: In years
- **Number of Siblings/Spouse (SibSp)**: Traveling together
- **Number of Parents/Children (Parch)**: On board
- **Fare**: Ticket price in British pounds
- **Port of Embarkation (Embarked)**: C = Cherbourg, Q = Queenstown, S = Southampton
""")

