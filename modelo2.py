# entrenar_cnn.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 15

if __name__ == "__main__":
    # 1) Cargar dataset desde carpeta
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    # Mejor rendimiento: cache & prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2) Definir modelo
    normalization_layer = layers.Rescaling(1./255)

    model = models.Sequential([
        normalization_layer,
        layers.Conv2D(16, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 3) Entrenar
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # 4) Guardar
    model.save("modelo_manzanas_cnn.h5")
    print("Modelo guardado en modelo_manzanas_cnn.h5")
