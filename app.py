from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load your fine-tuned PyTorch NLP model
tokenizer = AutoTokenizer.from_pretrained("clicknext/phayathaibert")
model = AutoModelForSequenceClassification.from_pretrained(
    "pytbert_sentiment_analysis")

# Set the model to evaluation mode
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the request
    data = request.get_json()
    text = data['text']

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class
    predicted_class = torch.argmax(outputs.logits).item()

    return jsonify({'predicted_class': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)
