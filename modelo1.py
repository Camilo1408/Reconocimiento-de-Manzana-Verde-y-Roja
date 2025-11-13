# entrenar_rf.py
import cv2
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

BINS = 16

def extraer_histograma(path, bins=BINS):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hist_r = cv2.calcHist([img], [0], None, [bins], [0,256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [bins], [0,256]).flatten()
    hist_b = cv2.calcHist([img], [2], None, [bins], [0,256]).flatten()

    hist = np.concatenate([hist_r, hist_g, hist_b])
    hist = hist / hist.sum()   # normalizar
    return hist

def cargar_dataset(base_dir):
    X, y = [], []
    for label, clase in enumerate(["roja", "verde"]):
        carpeta = os.path.join(base_dir, clase, "*.jpg")
        for path in glob.glob(carpeta):
            X.append(extraer_histograma(path))
            y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # 1) Cargar datos de train y test
    X_train, y_train = cargar_dataset("data/train")
    X_test, y_test = cargar_dataset("data/test")

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 2) Definir y entrenar modelo
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3) Evaluar
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy RF:", acc)
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=["roja", "verde"]))

    # 4) Guardar modelo
    joblib.dump(model, "modelo_manzanas_rf.pkl")
    print("Modelo guardado en modelo_manzanas_rf.pkl")
