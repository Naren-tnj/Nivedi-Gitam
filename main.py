from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras.layers import TFSMLayer
from tensorflow import keras

app = FastAPI()
MODEL = keras.layers.TFSMLayer("C:\\Projects\\Nivedi-Poultry\\saved_models\\1", call_endpoint='serving_default')
CLASS_NAMES=['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    batch_prediction = MODEL(img_batch)
    first_prediction = batch_prediction['output_0'][0]
    predicted_class_index = np.argmax(first_prediction)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence=np.max(first_prediction)
    return {"class": predicted_class, "confidence": float(confidence)}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)