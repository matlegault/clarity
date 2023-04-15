# -*- coding: utf-8 -*-
import argparse
import torch
import os
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template
from io import BytesIO
import base64



app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_image(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    Transform = transforms.Compose([
            transforms.Resize([224,224]),      
            transforms.ToTensor(),
            ])
    image = Transform(image)   
    image = image.unsqueeze(0)
    return image.to(device)

def predict(image, model):
    image = prepare_image(image)
    with torch.no_grad():
        preds = model(image)
    return preds.item()

@app.route("/")
def home():
    return render_template("/index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    image = Image.open(file)
    model = torchvision.models.resnet50()
    # model.avgpool = nn.AdaptiveAvgPool2d(1) # for any size of the input
    model.fc = torch.nn.Linear(in_features=2048, out_features=1)
    model.load_state_dict(torch.load('model/model-resnet50.pth', map_location=device)) 
    model.eval().to(device)
    score = predict(image, model)

    # Convert the image to a byte stream
    with BytesIO() as buffer:
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()

    image_data = base64.b64encode(image_bytes).decode('utf-8')

    return render_template('upload.html', score=score, image_data=image_data)

if __name__ == "__main__":
    app.run(debug=True)
