import matplotlib.pyplot as plt
import sys
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
from keras import backend as K

# function to plot the confusion matrix
def plot_confusion_matrix(Y_true, Y_predicted, classes, normalize=False):
    cm = confusion_matrix(Y_true, Y_predicted, labels = classes)
    plt.figure()
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.get_cmap('gray_r'))
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],  horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
# function to visualize 2D CNN filters    
def visualize_filter(model, input_X, input_Y, layer_num):
    if type(model.layers[layer_num]) != keras.layers.convolutional.Conv2D:
        sys.exit('The selected layer is not a 2D CNN filter') 
    get_filter_output = K.function([model.layers[0].input], [model.layers[layer_num].output])
    for k in range(0,10):
        X_layer = np.mean(input_X[np.where(input_Y == k)], axis=0)
        X_layer = np.expand_dims(X_layer, axis=0)
        X_layer = np.expand_dims(X_layer, axis=4)
        layer_output = get_filter_output([X_layer])[0]
        plt.figure(figsize=(20,20))
        for i in range(0, model.layers[k].get_config()['filters']):
            plt.subplot(1, 5, i+1)
            plt.title('Output of filter {0}, input = {1}'.format(i+1, k))
            plt.imshow(layer_output[0][:,:,i], cmap = plt.get_cmap('gray_r'))