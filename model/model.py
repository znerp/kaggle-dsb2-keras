### Model as of 06.12.2018

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    """
    RMSE loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)


def get_model():
    modelv01 = Sequential() # instantiate model with linear stack of layers (empty so far)
    modelv01.add(Activation(activation=center_normalize, input_shape=(30, 64, 64))) # should the input shape not be freely choosable?

    # CONV -> CONV -> MAXPOOL (previously padding) -> DROPOUT
    modelv01.add(Conv2D(64, kernel_size= (3, 3), padding='same'))
    modelv01.add(Activation('relu'))
    modelv01.add(Conv2D(64, kernel_size= (3, 3), padding='valid'))
    modelv01.add(Activation('relu'))
    modelv01.add(ZeroPadding2D(padding=(1, 1)))
    modelv01.add(MaxPooling2D(pool_size=(2, 2))) # stride is by defauls pool size in maxpool layers
    modelv01.add(Dropout(0.25))

    # CONV -> CONV -> MAXPOOL (previously padding) -> DROPOUT
    modelv01.add(Conv2D(96, kernel_size= (3, 3), padding='same'))
    modelv01.add(Activation('relu'))
    modelv01.add(Conv2D(96, kernel_size= (3, 3), padding='valid'))
    modelv01.add(Activation('relu'))
    modelv01.add(ZeroPadding2D(padding=(1, 1)))
    modelv01.add(MaxPooling2D(pool_size=(2, 2)))
    modelv01.add(Dropout(0.25))

    # CONV -> CONV -> MAXPOOL -> DROPOUT
    modelv01.add(Conv2D(128, kernel_size= (2, 2), padding='same'))
    modelv01.add(Activation('relu'))
    modelv01.add(Conv2D(128, kernel_size= (2, 2), padding='same'))
    modelv01.add(Activation('relu'))
    modelv01.add(MaxPooling2D(pool_size=(2, 2)))
    modelv01.add(Dropout(0.25))

    # FC -> DROPOUT -> FC
    modelv01.add(Flatten())
    modelv01.add(Dense(1024, kernel_regularizer=l2(1e-3)))
    modelv01.add(Activation('relu'))
    modelv01.add(Dropout(0.5))
    modelv01.add(Dense(1))

    adam = Adam(lr=0.0001)
    modelv01.compile(optimizer=adam, loss=root_mean_squared_error)
    return modelv01
