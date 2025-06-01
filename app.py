import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import base64

# Page config
st.set_page_config(page_title="Sonar Rock vs Mine Predictor", layout="centered")

# Load and train
@st.cache_data
def load_data_and_models():
    data = pd.read_csv("Copy of sonar data.csv", header=None)
    X = data.drop(columns=60)
    Y = data[60]

    # Classification Model
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.1, random_state=1)
    clf_model = LogisticRegression(max_iter=1000)
    clf_model.fit(X_train, Y_train)

    train_acc = accuracy_score(clf_model.predict(X_train), Y_train)
    test_acc = accuracy_score(clf_model.predict(X_test), Y_test)

    # Linear Regression Model (just for graph)
    Y_num = Y.map({'R': 0, 'M': 1})  # Convert classes to numbers
    lin_model = LinearRegression()
    lin_model.fit(X, Y_num)
    lin_pred = lin_model.predict(X)
    mse = mean_squared_error(Y_num, lin_pred)

    rock_mean = data[data[60] == 'R'].drop(columns=60).mean()
    mine_mean = data[data[60] == 'M'].drop(columns=60).mean()

    return data, clf_model, lin_model, train_acc, test_acc, mse, lin_pred, rock_mean, mine_mean

# Load everything
data, clf_model, lin_model, train_acc, test_acc, mse, lin_pred, rock_mean, mine_mean = load_data_and_models()

st.title("🔍 Sonar Rock vs Mine Predictor")

with st.expander("ℹ️ What are Rock vs Mine signals?"):
    st.markdown("""
    - **Rock (R):** Random, weaker sonar signal reflections.
    - **Mine (M):** Stronger, consistent patterns due to metal.
    - We use 60 sonar signal features and ML models to predict the object type.
    """)

# Accuracy Display
st.success(f"✅ Logistic Regression Accuracy — Train: `{train_acc:.2f}` | Test: `{test_acc:.2f}`")

# Correlation Matrix
with st.expander("📊 Correlation Matrix of Features"):
    st.subheader("Feature Correlation Heatmap")
    corr_matrix = data.drop(columns=60).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Linear Regression Graph
with st.expander("📈 Linear Regression Graph"):
    st.subheader("Actual vs Predicted Signal Class (0=Rock, 1=Mine)")
    fig2, ax2 = plt.subplots()
    ax2.scatter(range(len(data)), data[60].map({'R': 0, 'M': 1}), color='blue', label="Actual")
    ax2.plot(range(len(data)), lin_pred, color='red', label="Predicted")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Class")
    ax2.legend()
    ax2.set_title(f"Linear Regression Fit | MSE: {mse:.4f}")
    st.pyplot(fig2)

# Input Area
st.subheader("🔢 Input Sonar Signal (60 Features)")

# Initialize session state to persist values
if 'input_values' not in st.session_state:
    st.session_state.input_values = [0.0] * 60

# Random autofill button
if st.button("🎲 Auto-fill with random real values"):
    rand_row = data.sample(n=1).drop(columns=60).values.flatten().tolist()
    st.session_state.input_values = rand_row

# Input grid
cols = st.columns(3)
for i in range(60):
    st.session_state.input_values[i] = cols[i % 3].number_input(
        f"Feature {i+1}", min_value=0.0, max_value=1.0, step=0.01,
        value=float(st.session_state.input_values[i]), key=f"f_{i}"
    )

# Predict Button
if st.button("🔍 Predict Rock or Mine"):
    input_array = np.array(st.session_state.input_values).reshape(1, -1)
    prediction = clf_model.predict(input_array)[0]

    st.subheader("🎯 Prediction Result")
    if prediction == 'R':
        st.success("🪨 The object is predicted to be a **Rock**.")
    else:
        st.success("💣 The object is predicted to be a **Mine**.")

    # ✅ FIXED: Convert mean Series to NumPy arrays before subtraction
    rock_dist = np.linalg.norm(input_array - rock_mean.values)
    mine_dist = np.linalg.norm(input_array - mine_mean.values)

    st.write(f"📏 Distance to Rock mean: `{rock_dist:.2f}`")
    st.write(f"📏 Distance to Mine mean: `{mine_dist:.2f}`")

    if rock_dist < mine_dist:
        st.info("The input is **more similar to a Rock**.")
    else:
        st.info("The input is **more similar to a Mine**.")

    # Top contributing features
    coefs = clf_model.coef_[0]
    top_idx = np.argsort(np.abs(coefs))[-5:][::-1]
    st.subheader("🔝 Top 5 Influential Features")
    for i in top_idx:
        st.write(f"• Feature {i+1}: value = `{st.session_state.input_values[i]:.3f}`, weight = `{coefs[i]:.3f}`")

    # Signal visualization
    st.subheader("📈 Input Signal Plot")
    fig3, ax3 = plt.subplots()
    ax3.plot(range(1, 61), st.session_state.input_values, marker='o', color='teal')
    ax3.set_xlabel("Feature Number")
    ax3.set_ylabel("Signal Strength")
    ax3.set_title("Input Sonar Signal Pattern")
    st.pyplot(fig3)