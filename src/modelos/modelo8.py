

from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Activation, MaxPooling2D, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
from keras.optimizers import Adam
from keras.applications import EfficientNetB0


def estructura_modelo(input_shape=(128, 128, 3), num_classes=6, use_augmentation=False):
    
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = True

    model = Sequential()
    model.add(Input(shape=input_shape))

    if use_augmentation:
        model.add(RandomFlip("horizontal"))
        model.add(RandomRotation(0.15))
        model.add(RandomZoom(0.1))
        model.add(RandomContrast(0.1))
        model.add(RandomBrightness(0.1))

    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, use_bias=False))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, use_bias=False))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, use_bias=False))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    
    
    model.add(Dense(num_classes, activation='softmax'))

    model.build((None, *input_shape))
    model.summary()

    return model


def compilar(model, learning_rate=0.001):

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    return model
