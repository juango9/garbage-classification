"""
Modelo de red neuronal mas simple posible para la clasificacion de basura.

Este modelo constara de una sola capa oculata convolucional donde el numero de neuronas sera igual
al numero de clases, que tiene el conjunto de datos un total de 6 clases.

Una vez aplicada la funcion de activacion Relu se realizara un GlobalAvaregePooling 
o un GlobalMaxPooling para asi reducir cada una  de las matrices resultantes de 126x126 
matriz de 1x1, dando como resultado final con las 6 neuronas un vector de 1x6.

Luego se aplicara una capa densa con la funcion de activacion softmax para obtener las probabilidades de cada clase y asi realizar la clasificacion.
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPooling2D,GlobalAveragePooling2D
from keras import regularizers
from keras.optimizers import Adam


def estructura_modelo(input_shape=(128, 128, 3), num_classes=6):
    
    model = Sequential()

    model.add(Conv2D(num_classes, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(num_classes, activation='softmax'))

    resumen = model.summary()
    
    return model,resumen

def compilar(model,learning_rate):

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model

