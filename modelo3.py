import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (160, 160)  # Tamaño recomendado para MobileNetV2
BATCH_SIZE = 16
EPOCHS = 10

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

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(200).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # 2) Cargar MobileNetV2 preentrenado
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,   # quitamos la parte de clasificación original
        weights="imagenet"
    )
    base_model.trainable = False  # Lo usamos como extractor de características

    # 3) Construir el modelo final
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = preprocess_input(inputs)  # Normalización propia de MobileNet
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # 4) Entrenar
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # 5) Guardar modelo
    model.save("modelo_manzanas_tl.h5")
    print("Modelo TL guardado en modelo_manzanas_tl.h5")
