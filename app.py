from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("clicknext/phayathaibert")
model = AutoModelForSequenceClassification.from_pretrained(
    "Chonkator/feedback_topic_classifier")

# Set the model to evaluation mode
model.eval()

# Check if GPU is available
if torch.cuda.is_available():
    # Find the GPU with the highest compute capability
    device_id = torch.cuda.current_device()
    torch.cuda.set_device(device_id)
    device = torch.device("cuda")
    print(f"Using GPU: {device_id}")
else:
    print("No GPU available. Terminating the program.")
    exit()


def predict_labels(sentences):
    # Tokenize input sentences
    inputs = tokenizer(sentences, return_tensors="pt",
                       padding=True, truncation=True)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted labels
    predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()
    return predicted_labels


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Check if the file format is supported
    if not file.filename.endswith(('.csv', '.xlsx')):
        return jsonify({'error': 'Unsupported file format'})

    # Read the file
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # Get the filename without extension
    filename = os.path.splitext(file.filename)[0]

    # Export non-empty and non-NaN sentences to a text file
    input_filename = f"{filename}.txt"
    with open(input_filename, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            # Assuming the sentences are in the first column
            sentence = row.iloc[8]
            if pd.notna(sentence) and sentence.strip():  # Skip NaN and empty sentences
                f.write("%s\n" % sentence)

    # Tokenize sentences
    with open(input_filename, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]

    # Predict labels
    predicted_labels = predict_labels(sentences)

    # Write predicted labels to an output text file
    output_filename = f"{filename}_predicted_labels.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        for sentence, label in zip(sentences, predicted_labels):
            f.write(f"{sentence}\t{label}\n")

    return render_template('download.html', filename=output_filename, original_filename=filename)


@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
