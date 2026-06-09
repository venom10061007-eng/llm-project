# predict.py — Run a single prediction from a saved model
# Student  : مهند الحربي
# Instructor: المهندس راشد العقيل
# Usage    : python src/predict.py <image_path.png>

import sys
import numpy as np
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def predict_digit(image_path):
    model = keras.models.load_model('models/mnist_cnn.h5')
    img = plt.imread(image_path)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = 1.0 - img
    img = img / img.max()
    from PIL import Image
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    img_pil = img_pil.resize((28, 28), Image.LANCZOS)
    img = np.array(img_pil) / 255.0
    img = img[None, ..., None]
    probs = model.predict(img, verbose=0)[0]
    predicted = np.argmax(probs)
    confidence = probs[predicted] * 100
    print(f"Predicted digit : {predicted}")
    print(f"Confidence      : {confidence:.1f}%")
    return predicted, confidence

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path.png>")
    else:
        predict_digit(sys.argv[1])
