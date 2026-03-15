

from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout,Activation, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam


def estructura_modelo(input_shape=(128, 128, 3), num_classes=6):

    model = Sequential()

   
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    

  
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    


    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
  

    
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    return model


def compilar(model, learning_rate=0.001):

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    return model
