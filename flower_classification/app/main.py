from io import BytesIO

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image


def load_model():
    print('loading saved model ....')
    model = tf.keras.models.load_model('../model/flower_model')
    print('model loaded!')
    return model


# app
app = FastAPI()

# load model
model = load_model()
model.summary()


@app.get('/')
def home():
    return {"message": "Flower Classification API"}


@app.post('/predict')
async def predict_flower(file: UploadFile = File(...)):
    labels = ['dandelion', 'daisy', 'sunflower', 'tulip', 'rose']
    image = Image.open(BytesIO(await file.read()))
    image = np.asarray(image.resize((180, 180)))
    image = np.expand_dims(image, 0)
    image = image / 255
    prediction = model.predict(image)
    print(prediction)
    prediction = labels[np.argmax(prediction)]
    return {"Prediction": prediction}


if __name__ == "__main__":
    uvicorn.run('main:app', reload=True)
