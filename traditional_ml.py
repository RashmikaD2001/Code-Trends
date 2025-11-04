import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# Load the model and artifacts from Hugging Face (run once at the top)
model_path = hf_hub_download(repo_id="Juan12Dev/heart-risk-ai", filename="heart_risk_ensemble_v3.pkl")
model_data = joblib.load(model_path)

ensemble_model = model_data['ensemble_model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
optimal_threshold = model_data['optimal_threshold']


def create_feature_engineering(patient_data):
    """Apply same feature engineering steps used during model training."""
    df = pd.DataFrame([patient_data])

    # Convert and derive features
    df['age_in_days'] = df['age'] * 365.25
    df.rename(columns={'age_in_days': 'age'}, inplace=True)

    df['age_group'] = pd.cut(df['age']/365.25, bins=[0, 45, 55, 65, 100],
                             labels=['<45', '45-55', '55-65', '65+'])
    df['age_normalized'] = (df['age'] - (25 * 365.25)) / ((70*365.25) - (25*365.25))
    df['age_risk_exponential'] = np.where(df['age']/365.25 > 45,
                                          np.exp((df['age']/365.25 - 45) / 10), 1.0)

    df['bp_category'] = pd.cut(df['ap_hi'], bins=[0, 120, 140, 160, 180, 1000],
                               labels=['Normal', 'Elevated', 'Hypertension Stage 1',
                                       'Hypertension Stage 2', 'Hypertensive Crisis'])
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

    df['metabolic_profile'] = df['cholesterol'] / (df['age']/365.25)
    df['chol_category'] = pd.cut(df['cholesterol'], bins=[0, 1, 2, 3, 1000],
                                 labels=['Normal', 'Above Normal', 'Well Above Normal', 'High'])
    df['metabolic_syndrome_risk'] = ((df['cholesterol'] > 1).astype(int) +
                                     (df['gluc'] > 1).astype(int) +
                                     (df['ap_hi'] > 140).astype(int))

    df['male_age_interaction'] = df['gender'] * (df['age']/365.25)
    df['female_chol_interaction'] = (1 - df['gender']) * df['cholesterol']
    df['gender_specific_risk'] = np.where(df['gender'] == 2,
                                          (df['age']/365.25) * 0.1 + df['cholesterol'] * 0.005,
                                          df['cholesterol'] * 0.008)

    df['traditional_risk_score'] = (df['age']/365.25 * 0.04 + df['gender'] * 10 +
                                   (df['cholesterol'] - 1) * 20 + df['ap_hi'] * 0.1 +
                                   df['gluc'] * 20)
    df['cardiac_risk_score'] = (df['pulse_pressure'] * 0.2 + df['ap_hi'] * 0.1)
    df['combined_risk_score'] = (df['traditional_risk_score'] * 0.4 +
                                 df['cardiac_risk_score'] * 0.6)

    # Encode categories
    for col in ['age_group', 'chol_category', 'bp_category']:
        categories = ['<45', '45-55', '55-65', '65+', 'Normal', 'Elevated',
                      'Hypertension Stage 1', 'Hypertension Stage 2',
                      'Hypertensive Crisis', 'Above Normal', 'Well Above Normal', 'High']
        df[f'{col}_encoded'] = pd.Categorical(df[col], categories=categories).codes

    return df


def predict_heart_disease(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    """
    Predicts heart disease risk based on patient parameters.
    Returns probability and binary prediction (0=No, 1=Disease)
    """

    # Prepare patient input
    patient_data = {
        'age': age,           # years
        'gender': gender,     # 1: female, 2: male
        'height': height,     # cm
        'weight': weight,     # kg
        'ap_hi': ap_hi,       # systolic BP
        'ap_lo': ap_lo,       # diastolic BP
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active
    }

    # Apply feature engineering
    engineered_df = create_feature_engineering(patient_data)
    X = engineered_df.reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(X)

    # Predict probability
    probability = ensemble_model.predict_proba(X_scaled)[0, 1]
    prediction = int(probability >= optimal_threshold)

    print(f"Probability of Heart Disease: {probability:.2f}")
    print(f"Prediction (1=Disease, 0=No Disease): {prediction}")

    return {
        "probability": float(probability),
        "prediction": prediction
    }
