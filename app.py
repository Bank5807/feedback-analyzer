from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import os

app = Flask(__name__)


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
    export_filename = f"{filename}.txt"
    with open(export_filename, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            # Assuming the sentences are in the first column
            sentence = row.iloc[8]
            if pd.notna(sentence) and sentence.strip():  # Skip NaN and empty sentences
                f.write("%s\n" % sentence)

    return render_template('download.html', filename=export_filename, original_filename=filename)


@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
