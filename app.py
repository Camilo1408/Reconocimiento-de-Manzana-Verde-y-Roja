# app.py
from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
import os
from werkzeug.utils import secure_filename

BINS = 16

app = Flask(__name__)
model = joblib.load("modelo_manzanas_rf.pkl")

def extraer_histograma(path, bins=BINS):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hist_r = cv2.calcHist([img], [0], None, [bins], [0,256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [bins], [0,256]).flatten()
    hist_b = cv2.calcHist([img], [2], None, [bins], [0,256]).flatten()

    hist = np.concatenate([hist_r, hist_g, hist_b])
    hist = hist / hist.sum()
    return hist.reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def index():
    pred = None
    if request.method == "POST":
        file = request.files["imagen"]
        filename = secure_filename(file.filename)
        filepath = os.path.join("static", filename)
        os.makedirs("static", exist_ok=True)
        file.save(filepath)

        X = extraer_histograma(filepath)
        y_pred = model.predict(X)[0]
        pred = "Manzana roja" if y_pred == 0 else "Manzana verde"

    return render_template("index.html", pred=pred)

if __name__ == "__main__":
    app.run(debug=True)
