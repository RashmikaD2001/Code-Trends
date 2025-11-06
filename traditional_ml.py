import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os

# ---------------- LOAD MODEL ---------------- #
load_dotenv()
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
model_path = hf_hub_download(
    repo_id="Juan12Dev/heart-risk-ai-v4",
    filename="advanced_heart_risk_model_20250713_151433.pkl",
    token=HF_ACCESS_TOKEN
)

model_data = joblib.load(model_path)
calibrated_ensemble = model_data["calibrated_ensemble"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]
optimal_threshold = model_data["optimal_threshold"]

print(f"Model loaded successfully")
print(f"Optimal threshold: {optimal_threshold}")

# Get expected features from scaler
try:
    scaler_features = list(scaler.feature_names_in_)
    print(f"Scaler expects {len(scaler_features)} features")
except AttributeError:
    print("Warning: Scaler doesn't have feature_names_in_ attribute")
    scaler_features = feature_names

# ---------------- FEATURE ENGINEERING ---------------- #
def create_advanced_feature_engineering(patient_data: dict, age_in_days=True) -> pd.DataFrame:
    """
    Creates ALL features that were created during training.
    
    Args:
        patient_data: Dictionary with patient data
        age_in_days: If True, input age is in days and will be converted to years
    """
    df = pd.DataFrame([patient_data.copy()])
    
    # Convert age from days to years if needed
    if age_in_days:
        age_years = float(df["age"].iloc[0]) / 365.25
        # Update the dataframe to use years (model was trained on years)
        df["age"] = age_years
    else:
        age_years = float(df["age"].iloc[0])
    
    df["age"] = df["age"].astype(float)
    
    # 1. Age features - all calculations use age in years
    df["age_normalized"] = (age_years - 25) / (70 - 25)
    df["age_risk_exponential"] = np.where(
        age_years > 45,
        np.exp(np.clip((age_years - 45) / 10, 0, 5)),
        1.0
    )
    df["age_squared"] = age_years ** 2
    df["age_log"] = np.log1p(age_years)
    
    # 2. Blood pressure features
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["mean_arterial_pressure"] = df["ap_lo"] + (df["pulse_pressure"] / 3)
    
    # 3. Metabolic features
    df["metabolic_profile"] = df["cholesterol"] / max(age_years, 1)
    df["metabolic_syndrome_risk"] = (
        (df["cholesterol"] > 1).astype(int) + 
        (df["gluc"] > 1).astype(int) + 
        (df["ap_hi"] > 140).astype(int)
    )
    
    # 4. Gender interaction features
    df["male_age_interaction"] = (df["gender"] == 2).astype(int) * age_years
    df["female_chol_interaction"] = (df["gender"] == 1).astype(int) * df["cholesterol"]
    df["gender_specific_risk"] = np.where(
        df["gender"] == 1,
        df["cholesterol"] * 0.008,
        age_years * 0.1 + df["cholesterol"] * 0.005
    )
    
    # 5. Medical risk scores
    df["framingham_score"] = (
        age_years * 0.04 + 
        (df["ap_hi"] - 120) * 0.02 + 
        df["cholesterol"] * 15
    )
    df["traditional_risk_score"] = (
        age_years * 0.04 + 
        df["gender"] * 10 + 
        (df["cholesterol"] - 1) * 20 + 
        df["ap_hi"] * 0.1 + 
        df["gluc"] * 20
    )
    df["cardiac_risk_score"] = df["pulse_pressure"] * 0.2 + df["ap_hi"] * 0.1
    df["combined_risk_score"] = (
        df["traditional_risk_score"] * 0.4 + 
        df["cardiac_risk_score"] * 0.6
    )
    
    # 6. Statistical aggregations
    key_features = ['age', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
    available_features = [f for f in key_features if f in df.columns]
    if len(available_features) >= 3:
        feature_data = df[available_features].astype(float)
        df["feature_mean"] = feature_data.mean(axis=1)
        df["feature_std"] = feature_data.std(axis=1)
        df["feature_median"] = feature_data.median(axis=1)
        df["feature_max"] = feature_data.max(axis=1)
        df["feature_min"] = feature_data.min(axis=1)
        df["feature_range"] = df["feature_max"] - df["feature_min"]
    
    # 7. Age-based categorical encoding
    if age_years < 45:
        df["age_group_encoded"] = 0
        df["age_risk_category"] = 0
    elif age_years < 55:
        df["age_group_encoded"] = 1
        df["age_risk_category"] = 1
    elif age_years < 65:
        df["age_group_encoded"] = 2
        df["age_risk_category"] = 2
    else:
        df["age_group_encoded"] = 3
        df["age_risk_category"] = 3
    
    # 8. Cholesterol category encoding
    chol_val = df["cholesterol"].iloc[0]
    if chol_val <= 1.5:
        df["chol_category_encoded"] = 0
    elif chol_val <= 2.5:
        df["chol_category_encoded"] = 1
    elif chol_val <= 3.5:
        df["chol_category_encoded"] = 2
    else:
        df["chol_category_encoded"] = 3
    
    # 9. Blood pressure category encoding
    bp_val = df["ap_hi"].iloc[0]
    if bp_val < 120:
        df["bp_category_encoded"] = 0
    elif bp_val < 140:
        df["bp_category_encoded"] = 1
    elif bp_val < 160:
        df["bp_category_encoded"] = 2
    elif bp_val < 180:
        df["bp_category_encoded"] = 3
    else:
        df["bp_category_encoded"] = 4
    
    # 10. Polynomial features
    df["poly_age ap_hi"] = age_years * df["ap_hi"]
    df["poly_age cholesterol"] = age_years * df["cholesterol"]
    df["poly_age ap_lo"] = age_years * df["ap_lo"]
    df["poly_age gluc"] = age_years * df["gluc"]
    df["poly_ap_hi cholesterol"] = df["ap_hi"] * df["cholesterol"]
    df["poly_ap_hi ap_lo"] = df["ap_hi"] * df["ap_lo"]
    df["poly_ap_hi gluc"] = df["ap_hi"] * df["gluc"]
    df["poly_ap_lo cholesterol"] = df["ap_lo"] * df["cholesterol"]
    
    return df

# ---------------- PREDICTION FUNCTION ---------------- #
def predict_heart_disease(age, gender, height, weight, ap_hi, ap_lo, 
                         cholesterol, gluc, smoke, alco, active, 
                         age_in_days=True):
    """
    Predicts heart disease risk using the trained model.
    
    Parameters:
    - age: Age in days (default) or years
    - gender: 1 = female, 2 = male
    - height: Height in cm
    - weight: Weight in kg
    - ap_hi: Systolic blood pressure
    - ap_lo: Diastolic blood pressure
    - cholesterol: 1 = normal, 2 = above normal, 3 = well above normal
    - gluc: Glucose level (1 = normal, 2 = above normal, 3 = well above normal)
    - smoke: 0 = no, 1 = yes
    - alco: Alcohol intake (0 = no, 1 = yes)
    - active: Physical activity (0 = no, 1 = yes)
    - age_in_days: Whether age is provided in days (True) or years (False)
    """
    patient_data = {
        "age": age, "gender": gender, "height": height, "weight": weight,
        "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol, 
        "gluc": gluc, "smoke": smoke, "alco": alco, "active": active
    }
    
    # Create ALL features
    engineered_df = create_advanced_feature_engineering(patient_data, age_in_days)
    
    # Prepare features in exact order expected by scaler
    X = engineered_df.reindex(columns=scaler_features, fill_value=0)
    
    # Ensure all columns are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    probability = calibrated_ensemble.predict_proba(X_scaled)[0, 1]
    prediction = int(probability >= optimal_threshold)
    
    # Determine risk category
    if probability < 0.20:
        risk_category = "Low"
    elif probability < 0.45:
        risk_category = "Moderate"
    elif probability < 0.70:
        risk_category = "High"
    else:
        risk_category = "Critical"
    
    return {
        "probability": float(probability),
        "prediction": prediction,
        "risk_category": risk_category,
        "risk_label": "High Risk" if prediction else "Low Risk"
    }


# ---------------- BATCH PREDICTION FOR EVALUATION ---------------- #
def batch_predict(df, age_in_days=True, verbose=True):
    """
    Process multiple rows efficiently
    
    Args:
        df: DataFrame with patient data
        age_in_days: Whether age column is in days
        verbose: Print progress
    
    Returns:
        Lists of predictions, probabilities, and true labels
    """
    predictions = []
    probabilities = []
    true_labels = []
    
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        if verbose and (idx + 1) % 100 == 0:
            print(f"Processing row {idx + 1}/{total_rows}...")
        
        try:
            result = predict_heart_disease(
                age=row['age'], 
                gender=row['gender'], 
                height=row['height'], 
                weight=row['weight'],
                ap_hi=row['ap_hi'], 
                ap_lo=row['ap_lo'], 
                cholesterol=row['cholesterol'], 
                gluc=row['gluc'],
                smoke=row['smoke'], 
                alco=row['alco'], 
                active=row['active'],
                age_in_days=age_in_days
            )
            
            predictions.append(result["prediction"])
            probabilities.append(result["probability"])
            true_labels.append(int(row['cardio']))
            
        except Exception as e:
            if verbose:
                print(f"\nWarning: Prediction failed on row {idx}: {e}")
            predictions.append(0)
            probabilities.append(0.0)
            true_labels.append(int(row['cardio']))
    
    if verbose:
        print(f"\nProcessed all {total_rows} rows")
    
    return predictions, probabilities, true_labels


# ---------------- USAGE EXAMPLE ---------------- #
if __name__ == "__main__":
    # Single prediction example
    print("\n=== Single Prediction Test ===")
    result = predict_heart_disease(
        age=19386,  # age in days (53 years)
        gender=1,   # female
        height=155,
        weight=59.5,
        ap_hi=120,
        ap_lo=85,
        cholesterol=1,
        gluc=1,
        smoke=0,
        alco=0,
        active=1,
        age_in_days=True
    )
    print(f"Prediction: {result}")