from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, concatenate 
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D

# Multilayer perceptron 
def simple_nn(n_class, n_col):
    # input shape
    input_shape = Input(shape = (n_col,))
    # fully connected hidden layers
    fc_layer = Dense(500, activation = 'relu')(input_shape)
    fc_layer = Dropout(0.5)(fc_layer)
    fc_layer = Dense(100, activation = 'relu')(fc_layer)
    fc_layer = Dropout(0.4)(fc_layer)
    fc_layer = Dense(25, activation = 'relu')(fc_layer)
    fc_layer = Dropout(0.3)(fc_layer)
    # output layer
    if n_class > 2:
        out = Dense(n_class, activation = 'softmax')(fc_layer)
    else:
        out = Dense(n_class, activation = 'sigmoid')(fc_layer)
    model = Model(input_shape, out)
    return model

#Multilayer 1D CNN
def simple_1Dcnn(n_class, n_col):
    # input shape
    input_shape = Input(shape=(n_col,1))
    # 1D cnn layer
    cnn_layer = Conv1D(nb_filter = 5, kernel_size = 15, strides = 1, activation = 'relu')(input_shape)
    cnn_layer = Dropout(0.5)(cnn_layer)
    cnn_layer = (MaxPooling1D(pool_size = 2))(cnn_layer)
    # reshaping for fully connected layers
    cnn_flat = Flatten()(cnn_layer)
    # fully connected hidden layers
    fc_layer = Dense(500, activation = 'relu')(cnn_flat)
    fc_layer = Dropout(0.5)(fc_layer)
    fc_layer = Dense(100, activation = 'relu')(fc_layer)
    fc_layer = Dropout(0.4)(fc_layer)
    # output layer
    if n_class > 2:
        out = Dense(n_class, activation = 'softmax')(fc_layer)
    else:
        out = Dense(n_class, activation = 'sigmoid')(fc_layer)
    model = Model(input_shape, out)
    return model

#Multilayer 2D CNN
def simple_2Dcnn(n_class, image_height, image_width):
    # input shape
    input_shape = Input(shape=(image_height, image_width, 1))
    # cnn layer
    cnn_layer = Conv2D(nb_filter = 5, kernel_size = 15, strides = 1, activation = 'relu')(input_shape)
    cnn_layer = Dropout(0.5)(cnn_layer)
    cnn_layer = (MaxPooling2D(pool_size = 2))(cnn_layer)
    # reshaping for fully connected layers
    cnn_flat = Flatten()(cnn_layer)
    # fully connected hidden layers
    fc_layer = Dense(500, activation = 'relu')(cnn_flat)
    fc_layer = Dropout(0.5)(fc_layer)
    fc_layer = Dense(100, activation = 'relu')(fc_layer)
    fc_layer = Dropout(0.4)(fc_layer)
    # output layer
    if n_class > 2:
        out = Dense(n_class, activation = 'softmax')(fc_layer)
    else:
        out = Dense(n_class, activation = 'sigmoid')(fc_layer)
    model = Model(input_shape, out)
    return model

#Multilayer Stacked 1D CNN
def stacked_1Dcnn(n_class, n_col):
    # input shape
    input_shape = Input(shape=(n_col,1))
    # 1D cnn layer
    cnn_layer = Conv1D(nb_filter = 5, kernel_size = 15, strides = 1, activation = 'relu')(input_shape)
    cnn_layer = Dropout(0.5)(cnn_layer)
    cnn_layer = (MaxPooling1D(pool_size = 2))(cnn_layer)
    cnn_layer = Conv1D(nb_filter = 4, kernel_size = 10, strides = 1, activation = 'relu')(input_shape)
    cnn_layer = Dropout(0.4)(cnn_layer)
    cnn_layer = (MaxPooling1D(pool_size = 2))(cnn_layer)
    cnn_layer = Conv1D(nb_filter = 4, kernel_size = 5, strides = 1, activation = 'relu')(input_shape)
    cnn_layer = Dropout(0.3)(cnn_layer)
    cnn_layer = (MaxPooling1D(pool_size = 2))(cnn_layer)
    # reshaping for fully connected layers
    cnn_flat = Flatten()(cnn_layer)
    # fully connected hidden layers
    fc_layer = Dense(500, activation = 'relu')(cnn_flat)
    fc_layer = Dropout(0.5)(fc_layer)
    fc_layer = Dense(100, activation = 'relu')(fc_layer)
    fc_layer = Dropout(0.4)(fc_layer)
    # output layer
    if n_class > 2:
        out = Dense(n_class, activation = 'softmax')(fc_layer)
    else:
        out = Dense(n_class, activation = 'sigmoid')(fc_layer)
    model = Model(input_shape, out)
    return model

#Multilayer Stacked 2D CNN
def stacked_2Dcnn(n_class, image_height, image_width):
    # input shape
    input_shape = Input(shape=(image_height, image_width, 1))
    # cnn layers
    cnn_layer = Conv2D(nb_filter = 5, kernel_size = 15, strides = 1, activation = 'relu')(input_shape)
    cnn_layer = Dropout(0.5)(cnn_layer)
    cnn_layer = (MaxPooling2D(pool_size = 2))(cnn_layer)
    cnn_layer = Conv2D(nb_filter = 4, kernel_size = 10, strides = 1, activation = 'relu')(input_shape)
    cnn_layer = Dropout(0.4)(cnn_layer)
    cnn_layer = (MaxPooling2D(pool_size = 2))(cnn_layer)
    cnn_layer = Conv2D(nb_filter = 4, kernel_size = 5, strides = 1, activation = 'relu')(input_shape)
    cnn_layer = Dropout(0.3)(cnn_layer)
    cnn_layer = (MaxPooling2D(pool_size = 2))(cnn_layer)
    # reshaping for fully connected layers
    cnn_flat = Flatten()(cnn_layer)
    # fully connected hidden layers
    fc_layer = Dense(500, activation = 'relu')(cnn_flat)
    fc_layer = Dropout(0.5)(fc_layer)
    fc_layer = Dense(100, activation = 'relu')(fc_layer)
    fc_layer = Dropout(0.4)(fc_layer)
    # output layer
    if n_class > 2:
        out = Dense(n_class, activation = 'softmax')(fc_layer)
    else:
        out = Dense(n_class, activation = 'sigmoid')(fc_layer)
    model = Model(input_shape, out)
    return model

#Multilayer Multitowered Stacked 2D CNN
def multitower_2Dcnn(n_class, image_height, image_width):
    # input shape
    input_shape = Input(shape=(image_height, image_width, 1))
    # cnn-tower 1
    tower1 = Conv2D(nb_filter = 4, kernel_size = 15, strides = 1, activation = 'relu')(input_shape)
    tower1 = Dropout(0.5)(tower1)
    tower1 = Flatten()(tower1)
    # cnn-tower 2
    tower2 = Conv2D(nb_filter = 4, kernel_size = 15, strides = 1, activation = 'relu')(input_shape)
    tower2 = Dropout(0.5)(tower2)
    tower2 = Flatten()(tower2)
    # cnn-tower 3
    tower3 = Conv2D(nb_filter = 4, kernel_size = 15, strides = 1, activation = 'relu')(input_shape)
    tower3 = Dropout(0.5)(tower3)
    tower3 = Flatten()(tower3)
    # reshaping for fully connected layers
    cnn_flat = concatenate([tower1, tower2, tower3], axis=1)
    # fully connected hidden layers
    fc_layer = Dense(500, activation = 'relu')(cnn_flat)
    fc_layer = Dropout(0.5)(fc_layer)
    fc_layer = Dense(100, activation = 'relu')(fc_layer)
    fc_layer = Dropout(0.4)(fc_layer)
    # output layer
    if n_class > 2:
        out = Dense(n_class, activation = 'softmax')(fc_layer)
    else:
        out = Dense(n_class, activation = 'sigmoid')(fc_layer)
    model = Model(input_shape, out)
    return model