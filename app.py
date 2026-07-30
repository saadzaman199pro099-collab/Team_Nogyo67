from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import os
import requests
import tensorflow as tf


app = Flask(__name__)


# Create folders automatically
os.makedirs("uploadimages", exist_ok=True)
os.makedirs("models", exist_ok=True)



# ============================
# Download and Load AI Model
# ============================

MODEL_PATH = "models/plant_disease_recog_model_pwp.keras"

MODEL_URL = "https://huggingface.co/alexgromes/Plant/resolve/main/plant_disease_recog_model_pwp.keras"


if not os.path.exists(MODEL_PATH):

    print("Downloading model...")

    response = requests.get(
        MODEL_URL
    )

    if response.status_code != 200:
        raise Exception("Failed to download model")

    with open(MODEL_PATH, "wb") as file:
        file.write(response.content)

    print("Model downloaded successfully")


print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully")



# Disease labels

label = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]



# Load disease information

with open(
    "plant_disease.json",
    "r"
) as file:

    plant_disease = json.load(file)




# Serve uploaded images

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):

    return send_from_directory(
        './uploadimages',
        filename
    )





@app.route('/')
def home():

    return render_template(
        'home.html'
    )





# Image preprocessing

def extract_features(image):

    image = tf.keras.utils.load_img(
        image,
        target_size=(160,160)
    )


    feature = tf.keras.utils.img_to_array(
        image
    )


    feature = np.array(
        [feature]
    )


    return feature





# Prediction

def model_predict(image):

    img = extract_features(
        image
    )


    prediction = model.predict(
        img
    )


    confidence = round(
        float(np.max(prediction)) * 100,
        2
    )


    prediction_label = plant_disease[
        prediction.argmax()
    ]


    return prediction_label, confidence





@app.route(
    '/upload/',
    methods=['POST','GET']
)

def uploadimage():

    if request.method == "POST":

        image = request.files['img']


        if image.filename == "":

            return render_template(
                'home.html',
                error="Please select an image"
            )



        filename = (
            f"temp_{uuid.uuid4().hex}_{image.filename}"
        )


        filepath = os.path.join(
            "uploadimages",
            filename
        )


        image.save(
            filepath
        )



        prediction, confidence = model_predict(
            filepath
        )



        return render_template(
            'home.html',
            result=True,
            imagepath="/" + filepath.replace("\\","/"),
            prediction=prediction,
            confidence=confidence
        )



    else:

        return redirect('/')







if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )