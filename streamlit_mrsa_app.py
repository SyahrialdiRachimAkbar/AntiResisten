import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration ---
MODEL_PATH = "best_mrsa_prediction_model.joblib" # Relative path for Streamlit sharing
UNIQUE_VALUES_PATH = "categorical_unique_values.txt" # Relative path
COMBINED_DATA_PATH = "combined_mrsa_hospital_data.csv" # Relative path for slider stats

# --- Load Model ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file 	'{MODEL_PATH}	' not found. Please ensure it's in the GitHub repository root along with the app script.")
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# --- Load Unique Values for Inputs ---
@st.cache_data
def load_unique_values():
    unique_vals = {}
    default_infections_predicted = ["0.0", "1.0", "2.0", "Not Available"]
    default_hospital_category = ["Unknown", "GENERAL ACUTE CARE HOSPITAL"]
    default_county = ["Unknown", "Alameda"]

    if not os.path.exists(UNIQUE_VALUES_PATH):
        st.warning(f"Unique values file 	'{UNIQUE_VALUES_PATH}	' not found. Using default dropdown values.")
        unique_vals["Infections_Predicted"] = default_infections_predicted
        unique_vals["Hospital_Category_RiskAdjustment"] = default_hospital_category
        unique_vals["County"] = default_county
        return unique_vals
    
    current_key = None
    values_list = []
    try:
        with open(UNIQUE_VALUES_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("--- ") and line.endswith(" ---"):
                    if current_key and values_list:
                        unique_vals[current_key] = sorted(list(set(values_list)))
                    current_key = line.split(" (")[0].replace("--- ", "").strip()
                    values_list = []
                elif line and not line.startswith("===") and not line.startswith("Unique values") and current_key:
                    if "truncated" not in line:
                         values_list.append(line)
            if current_key and values_list: # Add the last key
                unique_vals[current_key] = sorted(list(set(values_list)))
    except Exception as e:
        st.error(f"Error reading 	'{UNIQUE_VALUES_PATH}	': {e}. Using default dropdown values.")
        unique_vals["Infections_Predicted"] = default_infections_predicted
        unique_vals["Hospital_Category_RiskAdjustment"] = default_hospital_category
        unique_vals["County"] = default_county
        return unique_vals
        
    expected_keys = ["Infections_Predicted", "Hospital_Category_RiskAdjustment", "County"]
    for key in expected_keys:
        if key not in unique_vals or not unique_vals[key]:
            st.warning(f"Key 	'{key}	' missing or empty in 	'{UNIQUE_VALUES_PATH}	'. Using default values for this dropdown.")
            if key == "Infections_Predicted": unique_vals[key] = default_infections_predicted
            elif key == "Hospital_Category_RiskAdjustment": unique_vals[key] = default_hospital_category
            elif key == "County": unique_vals[key] = default_county
            else: unique_vals[key] = ["Unknown"]
    return unique_vals

# --- Load data for min/max slider values ---
@st.cache_data
def load_data_for_sliders(data_path=COMBINED_DATA_PATH):
    min_max_vals = {
        "Patient_Days": (0.0, 200000.0, 5000.0), # min, max, default/median
        "SIR": (0.0, 10.0, 1.0) # min, max, default/median
    }
    if not os.path.exists(data_path):
        st.warning(f"Data file for slider ranges 	'{data_path}	' not found. Sliders will use default hardcoded ranges.")
        return min_max_vals

    try:
        df = pd.read_csv(data_path, low_memory=False)
        features_for_stats = ['Patient_Days', 'SIR']
        df_cleaned = df.dropna(subset=features_for_stats).copy()

        if "Patient_Days" in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned["Patient_Days"]) and not df_cleaned["Patient_Days"].empty:
            df_cleaned.loc[:, "Patient_Days"] = pd.to_numeric(df_cleaned["Patient_Days"], errors='coerce')
            df_cleaned.dropna(subset=["Patient_Days"], inplace=True)
            if not df_cleaned["Patient_Days"].empty:
                min_max_vals["Patient_Days"] = (
                    float(df_cleaned["Patient_Days"].min()), 
                    float(df_cleaned["Patient_Days"].max()),
                    float(df_cleaned["Patient_Days"].median())
                )
        if "SIR" in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned["SIR"]) and not df_cleaned["SIR"].empty:
            df_cleaned.loc[:, "SIR"] = pd.to_numeric(df_cleaned["SIR"], errors='coerce')
            df_cleaned.dropna(subset=["SIR"], inplace=True)
            if not df_cleaned["SIR"].empty:
                 min_max_vals["SIR"] = (
                    float(df_cleaned["SIR"].min()), 
                    float(df_cleaned["SIR"].max()),
                    float(df_cleaned["SIR"].median())
                )
    except Exception as e:
        st.warning(f"Error processing 	'{data_path}	' for slider ranges: {e}. Using default hardcoded ranges.")
    return min_max_vals

# --- Initialize ---
model = load_model()
unique_values = load_unique_values()
min_max_slider_values = load_data_for_sliders()

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
            min_value=min_max_slider_values["Patient_Days"][0],
            max_value=min_max_slider_values["Patient_Days"][1],
            value=min_max_slider_values["Patient_Days"][2], 
            step=100.0, 
            help="Enter the total patient days for the facility."
        )
        
        infections_predicted_options = unique_values.get("Infections_Predicted", ["0.0"])
        default_ip_index = 0
        if infections_predicted_options:
            common_defaults_ip = [val for val in ["0.0", "1.0", "2.0"] if val in infections_predicted_options]
            if common_defaults_ip:
                default_ip_index = infections_predicted_options.index(common_defaults_ip[0])

        infections_predicted = st.selectbox(
            "Infections Predicted (Number of infections predicted by NHSN model)", 
            options=infections_predicted_options,
            index=default_ip_index,
            help="Select the number of infections predicted by the NHSN model for the facility."
        )

    with col2:
        sir_val = st.number_input(
            "Standardized Infection Ratio (SIR)", 
            min_value=min_max_slider_values["SIR"][0],
            max_value=min_max_slider_values["SIR"][1],
            value=min_max_slider_values["SIR"][2],
            step=0.01,
            help="Enter the SIR value for the facility. SIR compares actual infections to predicted."
        )
        
        county_options = unique_values.get("County", ["Unknown"])
        default_county_index = 0
        if county_options and "Alameda" in county_options:
            default_county_index = county_options.index("Alameda")

        county = st.selectbox(
            "County", 
            options=county_options,
            index=default_county_index,
            help="Select the county where the hospital is located."
        )

    hospital_category_options = unique_values.get("Hospital_Category_RiskAdjustment", ["Unknown"])
    default_hc_index = 0
    if hospital_category_options and "GENERAL ACUTE CARE HOSPITAL" in hospital_category_options:
        default_hc_index = hospital_category_options.index("GENERAL ACUTE CARE HOSPITAL")
        
    hospital_category = st.selectbox(
        "Hospital Category (for Risk Adjustment)", 
        options=hospital_category_options,
        index=default_hc_index,
        help="Select the hospital's risk adjustment category.",
        key="hospital_cat"
    )

    submit_button = st.form_submit_button(label="Get Prediction")

# --- Prediction Logic and Display ---
if submit_button and model:
    input_data = pd.DataFrame({
        "Patient_Days": [patient_days],
        "Infections_Predicted": [str(infections_predicted)], 
        "Hospital_Category_RiskAdjustment": [hospital_category],
        "County": [county],
        "SIR": [sir_val]
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
        
        st.markdown("---")
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
st.markdown("---")
st.markdown("Developed by Manus AI for demonstration purposes.")
st.markdown("Dataset Source: California Health and Human Services Open Data Portal (MRSA BSI Data)")

