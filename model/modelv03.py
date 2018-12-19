### 17.12.2018
# Model v03 -> segmentation network
# implemented variable input size

from keras.models import Model
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
import keras as ks


def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)


def block_down(input_tensor, filters, kernel_size=(3,3), padding='same', f_act='relu'):
    """
    One whole downsampling block consisting of max pooling (factor 2) followed by two 2D convolutions with subsequent activation.
    Uses nearest interpolation for starters.
    """
    X = MaxPooling2D(pool_size=(2, 2))(input_tensor) # stride is by default pool size in maxpool layers
    X = Conv2D(filters, kernel_size=kernel_size, padding=padding)(X)
    X = Activation(f_act)(X)
    X = Conv2D(filters, kernel_size=kernel_size, padding=padding)(X)
    X = Activation(f_act)(X)

    return X

def block_up(input_tensor, fwd_tensor, filters, kernel_size=(3,3), padding='same', f_act='relu'):
    """
    One whole block of the expanding path, consisting of up-convolution, concatenation and two 2D convolutions with subsequent activation.
    """
    # X = Conv2D(filters, kernel_size=(1,1))(input_tensor) # reduce number of channels of input for merging to work
    try:
        X = UpSampling2D(size=(2, 2))(input_tensor) 
        assert(X.shape[-2:] == fwd_tensor.shape[-2:]) # check that the tensors can actually be concatenated
        X = Concatenate(axis=1)([X, fwd_tensor])
    except AssertionError:
        print('AssertionError: Shapes of input tensor and forward tensor are {} and {}.'.format(X.shape, fwd_tensor.shape))

    X = Conv2D(filters, kernel_size=kernel_size, padding=padding)(X)
    X = Activation(f_act)(X)
    X = Conv2D(filters, kernel_size=kernel_size, padding=padding)(X)
    X = Activation(f_act)(X)

    return X

def segnet_single_slice(input_shape=(1,128,128), keep_prob=1):
    """
    Model version 03: 2D U-net for segmentation of a single slice.
    No batch normalization implemented so far.
    487K params (independent of input shape since it has no fully connected layer).

    Input parameters:
    input_shape: shape of input (1 x height x width)
    keep_prob: keep probability for the four Dropout layers

    """
    p_drop = 1-keep_prob
    nFilters = [16,32,64,128]

    # define placeholder for input
    X_input = ks.layers.Input(input_shape)

    X = Activation(activation=center_normalize)(X_input) 

    ### Downsampling
    X = Conv2D(filters=nFilters[0], kernel_size=(3,3), activation='relu', padding='same')(X)
    X_fwd_1 = Conv2D(filters=nFilters[0], kernel_size=(3,3), activation='relu', padding='same')(X)
    # print(X_fwd_1.shape)
    # (MAXPOOL --> CONV -> CONV-> DROPOUT)x3
    X = block_down(X_fwd_1, filters=nFilters[1])
    X_fwd_2 = Dropout(p_drop)(X)
    # print(X_fwd_2.shape)

    X = block_down(X_fwd_2, filters=nFilters[2])
    X_fwd_3 = Dropout(p_drop)(X)
    # print(X_fwd_3.shape)

    X = block_down(X_fwd_3, filters=nFilters[3])
    X = Dropout(p_drop)(X)
    # print(X.shape)

    ### Upsampling
    # (Upsamling--> CONV -> CONV-> DROPOUT)x3
    X = block_up(X, X_fwd_3, filters=nFilters[2])
    X = Dropout(p_drop)(X)
    # print(X.shape)
    X = block_up(X, X_fwd_2, filters=nFilters[1])
    X = Dropout(p_drop)(X)
    # print(X.shape)
    X = block_up(X, X_fwd_1, filters=nFilters[0])
    X = Dropout(p_drop)(X)
    # print(X.shape)

    # sigmoid layer
    X = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(X)

    model = Model(inputs = X_input, outputs = X, name='ss_segnet_v01')

    return model


# get a model summary
# test_model = segnet_single_slice((1,184,184))
# test_model.compile(optimizer = 'Adam', loss = 'binary_crossentropy')

# test_model2 = segnet_single_slice((1,128,128))
# test_model2.compile(optimizer = 'Adam', loss = 'binary_crossentropy')

# test_model.summary()
# test_model2.summary()

# from keras.models import model_from_json
# import json

# json_string = test_model.to_json()
# with open('test.json', 'w') as jf:
#     json.dump(json_string, jf)
