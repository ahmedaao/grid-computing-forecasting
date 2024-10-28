# Import packages
import os
import json
import requests
import streamlit as st
import pandas as pd

# Import src modules
from dotenv import load_dotenv
from grid_computing_forecasting import config, dataset

# Load environment variables from .env file
load_dotenv()

# Load pickle object for test API
df = dataset.load_pickle_file(
    os.path.join(config.root_dir, "app/backend/df_sample.pkl")
)

# Function for prediction page
def prediction_page():
    st.title("Grid Computing Forecasting - Prediction")

    st.write(
        """This basic application is a POC (Proof Of Concept) which takes
        a job as input and generates a prediction thanks to a Deep Learning model embedded
        into an API."""
    )

    st.sidebar.write("## Input Data :gear:")

    # Extract unique job_id
    unique_job_id = df.index.tolist()
    selected_job_id = st.sidebar.selectbox("Choose a job_id", unique_job_id)
    st.write(f"You selected job_id: {selected_job_id}. Here are its features:")

    # Get the selected job's features as a DataFrame
    selected_job_features = df.loc[selected_job_id].to_frame().T

    # Format integers without decimals and floats with two decimals
    for col in selected_job_features.columns:
        if pd.api.types.is_integer_dtype(selected_job_features[col]):
            selected_job_features[col] = selected_job_features[col].astype(int)
        elif pd.api.types.is_float_dtype(selected_job_features[col]):
            selected_job_features[col] = selected_job_features[col].map("{:.2f}".format)

    # Display the formatted table in Streamlit
    st.table(selected_job_features)

    result = {"job_id": int(selected_job_id)}

    # Serialize and send inference request to FastAPI
    endpoint = st.sidebar.selectbox("Choose the API endpoint", ["prediction"])

    # Define URL based on running environment
    url = f"http://127.0.0.1:8000/{endpoint}"

    headers = {"Content-Type": "application/json"}
    result_json = json.dumps(result)
    response = requests.post(url, headers=headers, data=result_json, timeout=120)

    # Retrieve result from FastAPI and display it
    if response.status_code == 200:
        prediction = response.json().get("prediction")
        st.write("Prediction: ", prediction)
    else:
        st.write("Error:", response.status_code)

# Function for HTML display page
def html_page():
    st.title("Exploratory Data Analysis")
    st.write("This page summarizes the statistical analyses performed on the dataset.")

    # Load and display the HTML file
    html_file_path = os.path.join(config.root_dir, "reports/exploratory-data-analysis.html")
    with open(html_file_path, "r") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=600, scrolling=True)

# Main function
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "EDA"])

    # Page selection
    if page == "Prediction":
        prediction_page()
    elif page == "EDA":
        html_page()

if __name__ == "__main__":
    main()
