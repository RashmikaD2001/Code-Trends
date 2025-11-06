import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# === OPTIMIZED: Use FP16 + GPU + Faster Generation ===
model_id = "nztinversive/llama3.2-1B-HeartDiseasePrediction"

# Optional: 4-bit quantization for even more speed (if you have bitsandbytes)
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4"
# )

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,        # Use FP16
    device_map="auto",                # Auto-place on GPU
    # quantization_config=quantization_config,  # Uncomment for 4-bit
    low_cpu_mem_usage=True
)

# Move to GPU explicitly if needed
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("CUDA not available!")

def prediction_by_llm(age, height, weight, gender, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, context=""):
    feature_prompt = f"""
    Patient data:
    Age: {age} days,
    Height: {height} cm,
    Weight: {weight} kg,
    Gender code: {gender} (1=female, 2=male),
    Systolic BP: {ap_hi},
    Diastolic BP: {ap_lo},
    Cholesterol level: {cholesterol} (1=normal, 2=above normal, 3=well above normal),
    Glucose level: {gluc} (1=normal, 2=above normal, 3=well above normal),
    Smoking: {smoke} (0 = no, 1 = yes),
    Alcohol intake: {alco} (0 = no, 1 = yes),
    Physical activity: {active} (0 = no, 1 = yes).
    """
    output_prompt = """Output only JSON: {"has_heart_disease": boolean, "reason": "brief clinical explanation"} JSON:"""
    prompt = feature_prompt + context + output_prompt

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Move input to GPU

    with torch.no_grad():  # Save memory
        outputs = model.generate(
            **inputs,
            max_new_tokens=700,        # Only generate up to 150 new tokens
            temperature=0.5,           # Lower = more deterministic
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the new part after input
    response = generated_text[len(prompt):].strip()
    return response

# === TEST ===
if __name__ == "__main__":
    result = prediction_by_llm(
        age=19386,      # ~53 years
        gender=1,       # female
        height=155,
        weight=59.5,
        ap_hi=120,
        ap_lo=85,
        cholesterol=1,
        gluc=1,
        smoke=0,
        alco=0,
        active=1
    )
    print(result)