import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Iris ML App", layout="wide")

# Load model and data
model = joblib.load("model.pkl")
df = pd.read_csv("data/iris.csv")

st.title("🌸 Iris Flower Classification App")
st.write("Predict Iris species using a trained Machine Learning model.")

# Sidebar
menu = st.sidebar.selectbox(
    "Navigation",
    ["Data Exploration", "Visualisation", "Model Prediction", "Model Performance"]
)

if menu == "Data Exploration":
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write(df.dtypes)

elif menu == "Visualisation":
    st.subheader("Interactive Visualisation")
    fig = px.scatter(
        df,
        x="sepal length (cm)",
        y="petal length (cm)",
        color="target",
        title="Sepal vs Petal Length"
    )
    st.plotly_chart(fig)

elif menu == "Model Prediction":
    st.subheader("Make a Prediction")

    sl = st.number_input("Sepal Length", 4.0, 8.0, 5.1)
    sw = st.number_input("Sepal Width", 2.0, 4.5, 3.5)
    pl = st.number_input("Petal Length", 1.0, 7.0, 1.4)
    pw = st.number_input("Petal Width", 0.1, 2.5, 0.2)

    if st.button("Predict"):
        features = np.array([[sl, sw, pl, pw]])
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features).max()

        classes = ["Setosa", "Versicolor", "Virginica"]
        st.success(f"Prediction: {classes[prediction]}")
        st.info(f"Confidence: {confidence:.2f}")

elif menu == "Model Performance":
    st.subheader("Model Info")
    st.write("Model: Random Forest Classifier")
    st.write("Accuracy ~ 95%+")
