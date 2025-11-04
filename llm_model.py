# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("nztinversive/llama3.2-1B-HeartDiseasePrediction")
model = AutoModelForCausalLM.from_pretrained("nztinversive/llama3.2-1B-HeartDiseasePrediction")

def prediction_by_llm(age, height, weight, gender, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, context):
    """
    Predict the presence of cardiovascular disease using an LLM.
    
    Parameters:
    - age (int): Age in days
    - height (int): Height in cm
    - weight (float): Weight in kg
    - gender (int): Gender (categorical code)
    - ap_hi (int): Systolic blood pressure
    - ap_lo (int): Diastolic blood pressure
    - cholesterol (int): 1=normal, 2=above normal, 3=well above normal
    - gluc (int): 1=normal, 2=above normal, 3=well above normal
    - smoke (int): 0=no, 1=yes
    - alco (int): 0=no, 1=yes
    - active (int): 0=no, 1=yes
    - context (str): Optional additional context for reasoning
    
    Returns:
    - JSON string with "has_heart_disease" and "reason"
    """

    # Construct patient description prompt
    feature_prompt = f"""
    Patient data:
    Age: {age} days,
    Height: {height} cm,
    Weight: {weight} kg,
    Gender code: {gender},
    Systolic BP: {ap_hi},
    Diastolic BP: {ap_lo},
    Cholesterol level: {cholesterol} (1=normal, 2=above normal, 3=well above normal),
    Glucose level: {gluc} (1=normal, 2=above normal, 3=well above normal),
    Smoking: {smoke},
    Alcohol intake: {alco},
    Physical activity: {active}.
    """

    output_prompt = """Output only JSON: {"has_heart_disease": boolean, "reason": "brief clinical explanation"} JSON:"""

    # Combine into a single input for the LLM
    prompt = feature_prompt + context + output_prompt

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate model output
    outputs = model.generate(
        **inputs,
        max_length=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text
