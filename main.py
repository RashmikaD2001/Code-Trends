from traditional_ml import predict_heart_disease
from llm_model import prediction_by_llm

def main():
    print("Heart Disease Risk Prediction\n")

    # Collect user input
    age = int(input("Enter age (years): "))
    gender = int(input("Enter gender (1=female, 2=male): "))
    height = int(input("Enter height (cm): "))
    weight = float(input("Enter weight (kg): "))
    ap_hi = int(input("Enter systolic BP: "))
    ap_lo = int(input("Enter diastolic BP: "))
    cholesterol = int(input("Enter cholesterol (1=normal, 2=above normal, 3=well above normal): "))
    gluc = int(input("Enter glucose (1=normal, 2=above normal, 3=well above normal): "))
    smoke = int(input("Do you smoke? (0=no, 1=yes): "))
    alco = int(input("Do you consume alcohol? (0=no, 1=yes): "))
    active = int(input("Are you physically active? (0=no, 1=yes): "))

    context = input("Provide any other detail about patient you need to provide")

    # Run Ensemble Model
    ensemble_result = predict_heart_disease(
        age=age, gender=gender, height=height, weight=weight,
        ap_hi=ap_hi, ap_lo=ap_lo, cholesterol=cholesterol, gluc=gluc,
        smoke=smoke, alco=alco, active=active
    )

    # Run LLM Model
    llm_output = prediction_by_llm(
        age=age*365,
        height=height, weight=weight, gender=gender, ap_hi=ap_hi, ap_lo=ap_lo,
        cholesterol=cholesterol, gluc=gluc, smoke=smoke, alco=alco, active=active,
        context=context
    )

    # Extract LLM probability
    llm_prob = 1.0 if '"has_heart_disease": true' in llm_output.lower() else 0.0

    # Combine results
    final_prob = (ensemble_result["probability"] + llm_prob) / 2.0
    final_pred = 1 if final_prob >= 0.8 else 0

    # Display results
    print("\nModel Results:")
    print(f"Ensemble Model Probability: {ensemble_result['probability']:.2f}")
    print(f"LLM Model Probability: {llm_prob:.2f}")
    print(f"Combined Probability: {final_prob:.2f}")
    print(f"Final Prediction: {'Heart Disease Risk' if final_pred == 1 else 'No Significant Risk'}")
    
    print("\nLLM Clinical Reasoning:")
    print(llm_output)

if __name__ == "__main__":
    main()
