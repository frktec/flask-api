from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Modeli ve tokenizer'ı yükleyin
model_name = "mesut/turkish_law_article_justifier_llama-3.1_lora_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route("/predict", methods=["POST"])
def predict():
    input_text = request.json.get("input_text", "")
    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
