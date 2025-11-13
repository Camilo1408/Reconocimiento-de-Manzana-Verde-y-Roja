🍎 Reconocimiento de Manzana Roja y Verde

Proyecto de clasificación de imágenes utilizando Machine Learning, Transfer Learning y un servicio web Flask para identificar si una manzana es roja o verde a partir de una fotografía.

🚀 Requisitos previos
🔧 Instalar Python 3.10.11

Descargar desde:
https://www.python.org/downloads/windows

🧱 1. Crear el entorno virtual
python3.10 -m venv .venv


Activar el entorno:

Windows:
.\venv\Scripts\activate

📦 2. Instalar dependencias
Opción A — Instalar paquete por paquete
pip install numpy opencv-python scikit-learn matplotlib joblib
pip install tensorflow
pip install flask

Opción B — Instalar todo desde requirements.txt (RECOMENDADO)
pip install -r requirements.txt

🧠 3. Entrenar los modelos

Ejecuta los scripts según el modelo:

python modelo1.py
python modelo2.py
python modelo3_tl.py


Cada archivo entrenará un modelo distinto y generará sus pesos correspondientes.

🌐 4. Ejecutar la aplicación Flask

Una vez entrenado el modelo:

python app.py


La aplicación iniciará en:

http://127.0.0.1:5000

Sube una imagen y el sistema detectará si la manzana es roja o verde.
