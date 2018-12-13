### 11.12.2018
# Model v02
# implemented variable input size

from keras.models import Model
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
import keras as ks


def RMSE(y_true, y_pred):
    """
    RMSE loss function 
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)


def get_model(input_shape, keep_prob=1, f_act='relu'):
    """
    Model version 02: almost VGG style (just shallower) -> systematic increase of depth 
    by factor of 2 (dimension 1), decrease of width and height by factor of 2 
    (dimension 2 and 3) only through max pooling after 2 convolutions with 'same' padding.
    Slightly below 7M trainable parameters; would be down to 680K if flatten layer could be removed.
    Input gets center_normalized.

    Input parameters:
    input_shape: shape of input (t x height x width)
    keep_prob: keep probability for the four Dropout layers
    f_act: activation function of the convolutional layers

    """
    p_drop = 1-keep_prob

    # define placeholder for input
    X_input = ks.layers.Input(input_shape)

    X = Activation(activation=center_normalize)(X_input) 

    # CONV -> CONV -> MAXPOOL (previously padding) -> DROPOUT
    X = Conv2D(filters = 64, kernel_size= (3, 3), padding='same')(X)
    X = Activation(f_act)(X)
    X = Conv2D(64, kernel_size= (3, 3), padding='same')(X)
    X = Activation(f_act)(X)
    X = MaxPooling2D(pool_size=(2, 2))(X) # stride is by defauls pool size in maxpool layers
    X = Dropout(p_drop)(X)

    # CONV -> CONV -> MAXPOOL (previously padding) -> DROPOUT
    X = Conv2D(128, kernel_size= (3, 3), padding='same')(X)
    X = Activation(f_act)(X)
    X = Conv2D(128, kernel_size= (3, 3), padding='same')(X)
    X = Activation(f_act)(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(p_drop)(X)

    # CONV -> CONV -> MAXPOOL -> DROPOUT
    X = Conv2D(256, kernel_size= (2, 2), padding='same')(X)
    X = Activation(f_act)(X)
    X = Conv2D(256, kernel_size= (2, 2), padding='same')(X)
    X = Activation(f_act)(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(p_drop)(X)

    # FC -> DROPOUT -> FC
    X = Flatten()(X) # could flatten be removed if the dense layer was at size 256?
    X = Dense(1024)(X)#, kernel_regularizer=l2(1e-3))(X)
    X = Activation(f_act)(X)
    X = Dropout(p_drop)(X)
    X = Dense(1)(X)

    model = Model(inputs = X_input, outputs = X, name='modelv02')

    return model


# # get a model summary
# test_model = get_model((30,64,64))
# test_model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy')
# test_model.summary()

