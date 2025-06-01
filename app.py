import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import base64

# Load the dataset
sonar_data = pd.read_csv("Copy of sonar data.csv", header=None)

# Prepare data
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Distance calculator
def calc_distances(input_values):
    rock_mean = X[Y == 'R'].mean().to_numpy()
    mine_mean = X[Y == 'M'].mean().to_numpy()
    rock_dist = np.linalg.norm(input_values - rock_mean)
    mine_dist = np.linalg.norm(input_values - mine_mean)
    return rock_dist, mine_dist

# Generate PDF
def generate_pdf(prediction, input_values, rock_dist, mine_dist):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp_file.name, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Sonar Rock vs Mine Prediction Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 90, f"Prediction Result: {'Rock' if prediction == 'R' else 'Mine'}")
    c.drawString(50, height - 110, f"Distance to Rock Mean: {rock_dist:.2f}")
    c.drawString(50, height - 130, f"Distance to Mine Mean: {mine_dist:.2f}")
    c.drawString(50, height - 160, "Input Signal Values:")

    for i in range(60):
        y = height - 180 - (i * 12)
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(70, y, f"Feature {i+1}: {input_values[i]:.3f}")

    c.save()
    return tmp_file.name

# Streamlit UI
st.set_page_config(page_title="Sonar Rock vs Mine Predictor")
st.title("🛰️ Sonar Rock vs Mine Predictor")

# Show correlation matrix
with st.expander("📊 Show Correlation Matrix"):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(sonar_data.corr(), ax=ax, cmap='coolwarm', annot=False)
    st.pyplot(fig)

# Linear Regression Visualization
with st.expander("📈 Linear Regression Visualization")):
    lin_reg = LinearRegression()
    lin_reg.fit(np.arange(60).reshape(-1, 1), X.iloc[0].values.reshape(-1, 1))
    y_pred = lin_reg.predict(np.arange(60).reshape(-1, 1))
    plt.figure(figsize=(8, 4))
    plt.plot(X.columns, X.iloc[0].values, label='Original')
    plt.plot(X.columns, y_pred, '--', label='Linear Regression Fit')
    plt.legend()
    st.pyplot(plt)

# Input fields
st.subheader("🔢 Enter or Generate Input Features")

if 'input_values' not in st.session_state:
    st.session_state.input_values = np.zeros(60)

cols = st.columns(6)
for i in range(60):
    with cols[i % 6]:
        st.session_state.input_values[i] = st.number_input(f"F{i+1}", value=float(st.session_state.input_values[i]))

# Generate random values button
if st.button("🎲 Generate Random Values"):
    st.session_state.input_values = X.sample(1).values.flatten()
    st.experimental_rerun()

# Prediction button
if st.button("🔍 Predict Rock or Mine"):
    input_reshaped = np.asarray(st.session_state.input_values).reshape(1, -1)
    prediction = model.predict(input_reshaped)[0]
    rock_dist, mine_dist = calc_distances(st.session_state.input_values)

    st.session_state.prediction = prediction
    st.session_state.rock_dist = rock_dist
    st.session_state.mine_dist = mine_dist

    st.write(f"### ✅ Prediction: The object is a {'Rock' if prediction == 'R' else 'Mine'}")
    st.write(f"- 📏 Distance to Rock Mean: {rock_dist:.2f}")
    st.write(f"- 📏 Distance to Mine Mean: {mine_dist:.2f}")

# Download PDF button
if "prediction" in st.session_state:
    if st.button("📥 Download Prediction as PDF"):
        pdf_path = generate_pdf(
            st.session_state.prediction,
            st.session_state.input_values,
            st.session_state.rock_dist,
            st.session_state.mine_dist
        )
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="prediction_report.pdf">📄 Click here to download your report</a>'
            st.markdown(href, unsafe_allow_html=True)
