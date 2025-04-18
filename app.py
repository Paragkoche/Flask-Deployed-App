import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
from gdown import download


disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

model = CNN.CNN(39)
if not os.path.exists("plant_disease_model_1_latest.pt"):
    url = "https://drive.google.com/uc?id=1GWJ5HC8LxQsExmHEZWf7q89MRUquqt7U"
    output = "plant_disease_model_1_latest.pt"
    download(url, output, quiet=False)

model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()


def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)


@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/contact')
def contact():
    return render_template('contact-us.html')


@app.route('/index')
def ai_engine_page():
    return render_template('index.html')


@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    image = request.files['file']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join('static/uploads', image.filename)
    image.save(file_path)

    pred = prediction(file_path)

    response = {
        'disease_name': disease_info['disease_name'][pred],
        'description': disease_info['description'][pred],
        'possible_steps': disease_info['Possible Steps'][pred],
        'image_url': disease_info['image_url'][pred],
        'supplement_name': supplement_info['supplement name'][pred],
        'supplement_image': supplement_info['supplement image'][pred],
        'buy_link': supplement_info['buy link'][pred],
        'confidence_level': 1.0  # Optional: replace with model confidence if available
    }

    return jsonify(response)


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = str(image.filename)
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name, simage=supplement_image_url, buy_link=supplement_buy_link)
    else:
        return "Error"


@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))


if __name__ == '__main__':
    app.run(debug=True)
