import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title=("Heart Disease Prediction"),
    layout="wide"
)

st.title("Heart Disease Prediction")
st.write(
    """
    This application evaluates multiple machine learning models for Heart Disease prediction.
    Users can select a trained model and upload a dataset to view performance metrics and confusion matrix.
    """
)

st.markdown("### Dataset Selection")
st.write("You may upload your own CSV dataset containing a 'target' column. If no file is uploaded, the application will automatically use the default heart.csv sample dataset.")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV test dataset (optional)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Custom dataset uploaded successfully. Evaluating model on uploaded data.")
else:
    st.warning("No dataset uploaded. Using default sample dataset (heart.csv) from the repository for evaluation.")
    data = pd.read_csv("heart.csv")

    if "target" in data.columns:
        X_test = data.drop("target", axis=1)
        y_test = data["target"]

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        st.subheader("Evaluation Metrics")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("AUC:", roc_auc_score(y_test, y_proba))
        st.write("Precision:", precision_score(y_test, y_pred))
        st.write("Recall:", recall_score(y_test, y_pred))
        st.write("F1 Score:", f1_score(y_test, y_pred))
        st.write("MCC:", matthews_corrcoef(y_test, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

    else:
        st.error("Uploaded CSV must contain a 'target' column.")

# Load trained models
models = {
    "Logistic Regression": joblib.load("model/logistic.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}

# Model selection
model_name = st.selectbox("Select a Machine Learning Model", list(models.keys()))
model = models[model_name]
