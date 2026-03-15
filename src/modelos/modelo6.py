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
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout,Activation, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications import ResNet50


def estructura_modelo(input_shape=(128, 128, 3), num_classes=6):
    
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = True

    model = Sequential()

    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    return model


def compilar(model, learning_rate=0.001):

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    return model
