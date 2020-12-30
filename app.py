import base64

import numpy as np
import io
# import cv2

from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.image import img_to_array


from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS

from pickle import dump, load


app = Flask(__name__)
cors = CORS(app)


def get_model():
    global model, tokenizer, features_vector, vgmodel

    model = load_model('ml/icg.h5')
    tokenizer = load(open("ml/tokenizer.p", "rb"))
    features_vector = load(open("ml/features.p", "rb"))
    vgmodel = VGG16()
    vgmodel.layers.pop()
    vgmodel = Model(inputs=vgmodel.inputs, outputs=vgmodel.layers[-2].output)
    print('model loaded')


get_model()


@app.route('/predict', methods=["POST"])
def predict():
    max_length = 36
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))

    feature = extract_features(image, vgmodel)
    prediction = generate_desc(model, tokenizer, feature, max_length)

    response = {
        'prediction': prediction.replace('startofseq', '').replace('endofseq', ''),
    }

    return jsonify(response)


def extract_features(image, model):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image)
    return feature


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, feature, max_length):
    in_text = 'startofseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")

        pred = model.predict([feature, sequence])
        pred = np.argmax(pred)

        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endofseq':
            break
    return in_text
