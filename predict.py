import cv2
import numpy as np
import tensorflow as tf

IMAGE_SIZE = (128, 128)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return img

def predict_pollution_level(model, image_path):
    img = preprocess_image(image_path)
    if img is None:
        return "Error processing image"
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    labels = ['Low', 'Medium', 'High']
    return labels[predicted_class]
