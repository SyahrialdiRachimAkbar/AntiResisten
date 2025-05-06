import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration ---
MODEL_PATH = "/home/ubuntu/best_mrsa_prediction_model.joblib"
UNIQUE_VALUES_PATH = "/home/ubuntu/categorical_unique_values.txt" # To load options for selectboxes

# --- Load Model and Unique Values for Inputs ---
@st.cache_resource # Cache the model loading
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved.")
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

@st.cache_data # Cache the unique values loading
def load_unique_values():
    unique_vals = {}
    if not os.path.exists(UNIQUE_VALUES_PATH):
        st.warning(f"Unique values file not found at {UNIQUE_VALUES_PATH}. Categorical dropdowns might be empty or use defaults.")
        # Provide some defaults if file is missing, or handle this more gracefully
        unique_vals["Infections_Predicted"] = ["0.0", "1.0", "2.0"] # Example default
        unique_vals["Hospital_Category_RiskAdjustment"] = ["Unknown"] # Example default
        unique_vals["County"] = ["Unknown"] # Example default
        return unique_vals
    
    current_key = None
    values_list = []
    try:
        with open(UNIQUE_VALUES_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("--- ") and line.endswith(" ---"):
                    if current_key and values_list:
                        unique_vals[current_key] = sorted(list(set(values_list))) # Ensure unique and sorted
                    current_key = line.split(" (")[0].replace("--- ", "").strip()
                    values_list = []
                elif line and not line.startswith("===") and not line.startswith("Unique values") and current_key:
                    if "truncated" not in line: # Avoid adding "... (truncated)"
                         values_list.append(line)
            if current_key and values_list: # Add the last key
                unique_vals[current_key] = sorted(list(set(values_list)))
        
        # Ensure all expected keys are present, even if empty from file parsing
        expected_keys = ["Infections_Predicted", "Hospital_Category_RiskAdjustment", "County"]
        for key in expected_keys:
            if key not in unique_vals:
                st.warning(f"Key 	hemed_mrsa_hospital_data.csv

# --- 1. Load Data ---
print(f"Loading data from {input_csv}...")
df = pd.read_csv(input_csv, low_memory=False)
print(f"Data loaded. Shape: {df.shape}")

# --- 2. Replicate Cleaning Steps from Training Script ---
if 'Comparison' not in df.columns:
    print("Target column 'Comparison' not found. Exiting.")
    exit()

df_model = df.copy()
df_model.dropna(subset=["Comparison"], inplace=True) # Remove rows where target is NaN

features_for_stats = [
    'Patient_Days',
    'Infections_Predicted',
    'Hospital_Category_RiskAdjustment',
    'County',
    'SIR'
]

# IMPORTANT: Drop rows where any of these key features are NaN (as done in training)
df_model_cleaned = df_model.dropna(subset=features_for_stats)

# --- Get min/max for numerical features for sliders ---
min_max_values = {}
if not df_model_cleaned.empty:
    min_max_values["Patient_Days"] = (float(df_model_cleaned["Patient_Days"].min()), float(df_model_cleaned["Patient_Days"].max()))
    min_max_values["SIR"] = (float(df_model_cleaned["SIR"].min()), float(df_model_cleaned["SIR"].max()))
else:
    # Fallback default values if dataframe is empty after cleaning (should not happen with good data)
    min_max_values["Patient_Days"] = (0.0, 100000.0) 
    min_max_values["SIR"] = (0.0, 5.0)

# --- App Layout ---
st.set_page_config(page_title="MRSA Infection Risk Predictor", layout="wide")
st.title("🔬 MRSA Infection Risk Predictor")
st.markdown("""
This application predicts the likelihood of a hospital having a 'Worse' than expected 
rate of Methicillin-resistant Staphylococcus aureus (MRSA) bloodstream infections (BSI) 
compared to a national baseline. 

Provide the following information for a hospital to get a prediction.
""")

# --- User Inputs in a Form ---
with st.form("prediction_form"):
    st.subheader("Hospital and Patient Data")
    col1, col2 = st.columns(2)

    with col1:
        patient_days = st.number_input(
            "Patient Days (Total number of days patients were in the hospital)", 
            min_value=min_max_values["Patient_Days"][0],
            max_value=min_max_values["Patient_Days"][1],
            value=float(df_model_cleaned["Patient_Days"].median()) if not df_model_cleaned.empty else 5000.0, 
            step=100.0, 
            help="Enter the total patient days for the facility."
        )
        
        # Infections_Predicted is categorical in the model due to how it was handled (string numbers)
        # It was originally numeric but became object type. The unique values file treats it as categorical.
        infections_predicted_options = unique_values.get("Infections_Predicted", ["0.0"])
        # Try to find a reasonable default, like the most frequent or median if possible
        default_ip = "0.0"
        if not df_model_cleaned.empty and "Infections_Predicted" in df_model_cleaned.columns:
            try:
                # Convert to string to match options if they are strings
                mode_val = str(df_model_cleaned["Infections_Predicted"].mode()[0])
                if mode_val in infections_predicted_options:
                    default_ip = mode_val
            except KeyError: # mode() can fail on empty or all-NaN series
                pass
        
        infections_predicted = st.selectbox(
            "Infections Predicted (Number of infections predicted by NHSN model)", 
            options=infections_predicted_options,
            index=infections_predicted_options.index(default_ip) if default_ip in infections_predicted_options else 0,
            help="Select the number of infections predicted by the NHSN model for the facility."
        )

    with col2:
        sir = st.number_input(
            "Standardized Infection Ratio (SIR)", 
            min_value=min_max_values["SIR"][0],
            max_value=min_max_values["SIR"][1],
            value=float(df_model_cleaned["SIR"].median()) if not df_model_cleaned.empty else 1.0,
            step=0.01,
            help="Enter the SIR value for the facility. SIR compares actual infections to predicted."
        )
        
        county_options = unique_values.get("County", ["Unknown"])
        default_county = "Alameda" # A common county, or use mode from data
        if not df_model_cleaned.empty and "County" in df_model_cleaned.columns:
            try:
                mode_val = df_model_cleaned["County"].mode()[0]
                if mode_val in county_options:
                    default_county = mode_val
            except KeyError:
                pass
        county = st.selectbox(
            "County", 
            options=county_options,
            index=county_options.index(default_county) if default_county in county_options else 0,
            help="Select the county where the hospital is located."
        )

    hospital_category_options = unique_values.get("Hospital_Category_RiskAdjustment", ["Unknown"])
    default_hc = "GENERAL ACUTE CARE HOSPITAL" # A common category
    if not df_model_cleaned.empty and "Hospital_Category_RiskAdjustment" in df_model_cleaned.columns:
        try:
            mode_val = df_model_cleaned["Hospital_Category_RiskAdjustment"].mode()[0]
            if mode_val in hospital_category_options:
                default_hc = mode_val
        except KeyError:
            pass 
    hospital_category = st.selectbox(
        "Hospital Category (for Risk Adjustment)", 
        options=hospital_category_options,
        index=hospital_category_options.index(default_hc) if default_hc in hospital_category_options else 0,
        help="Select the hospital's risk adjustment category.",
        key="hospital_cat"
    )

    submit_button = st.form_submit_button(label="Get Prediction")

# --- Prediction Logic and Display ---
if submit_button and model:
    # Create a DataFrame for the input features
    # Ensure the order and naming matches the training data
    input_data = pd.DataFrame({
        "Patient_Days": [patient_days],
        "Infections_Predicted": [str(infections_predicted)], # Ensure it's string as per model training
        "Hospital_Category_RiskAdjustment": [hospital_category],
        "County": [county],
        "SIR": [sir]
    })
    
    st.subheader("Prediction Results")
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        if prediction == 1:
            st.error("Prediction: Hospital is at risk of having a 'WORSE' MRSA infection rate.", icon="🚨")
        else:
            st.success("Prediction: Hospital is NOT at risk of having a 'WORSE' MRSA infection rate.", icon="✅")
        
        st.write(f"Probability of 'Worse' Rate: {prediction_proba[1]:.2%}")
        st.write(f"Probability of 'Not Worse' Rate: {prediction_proba[0]:.2%}")
        
        # Explanation
        st.markdown("--- ")
        st.markdown("**Understanding the Prediction:**")
        st.markdown(""" 
        - **'Worse' Rate:** Indicates that the number of observed MRSA bloodstream infections is statistically significantly higher than the number of predicted infections based on national data and facility characteristics.
        - **'Not Worse' Rate:** Indicates that the observed MRSA BSI rate is similar to or better than predicted.
        
        This prediction is based on a Gradient Boosting machine learning model trained on historical data from California hospitals. 
        It should be used as a supportive tool and not as a definitive diagnosis or assessment.
        """)
        
        with st.expander("View Input Data for this Prediction"):
            st.dataframe(input_data)
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

elif submit_button and not model:
    st.error("Model is not loaded. Cannot make a prediction.")

# --- Footer/Information ---
st.markdown("--- ")
st.markdown("Developed by Manus AI for demonstration purposes.")
st.markdown("Dataset Source: California Health and Human Services Open Data Portal (MRSA BSI Data)")


