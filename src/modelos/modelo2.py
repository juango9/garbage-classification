"""
Modelo de red neuronal convolucional con 4 bloques para la clasificacion de basura.

Arquitectura:
  - Bloque 1: 2x Conv2D(32)  + BatchNorm + MaxPooling + Dropout(0.25)  -> 64x64x32
  - Bloque 2: 2x Conv2D(64)  + BatchNorm + MaxPooling + Dropout(0.25)  -> 32x32x64
  - Bloque 3: 2x Conv2D(128) + BatchNorm + MaxPooling + Dropout(0.25)  -> 16x16x128
  - Bloque 4: 2x Conv2D(256) + BatchNorm + GlobalAveragePooling2D      -> 256
  - Dense(256, relu) + BatchNorm + Dropout(0.5)
  - Dense(num_classes, softmax)

El aumento progresivo de filtros permite capturar caracteristicas de baja y alta
frecuencia. BatchNormalization acelera la convergencia y Dropout evita el sobreajuste
al incrementar la capacidad del modelo respecto al Modelo 1.
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam


def estructura_modelo(input_shape=(128, 128, 3), num_classes=6):

    model = Sequential()

    # Bloque 1 — 32 filtros -> 64x64
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    #model.add(BatchNormalization())
    #model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    # Bloque 2 — 64 filtros -> 32x32
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    # Bloque 3 — 128 filtros -> 16x16
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    # Bloque 4 — 256 filtros + pooling global -> vector de 256
    #model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(GlobalAveragePooling2D())

    # Cabeza clasificadora
    model.add(GlobalAveragePooling2D())
    #model.add(Dense(256, activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    return model


def compilar(model, learning_rate=0.001):

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    return model
