# DeepLearn_NUMBERS
## A brief dive into the MNIST dataset.

The MNIST problem is a dataset developed by Yann LeCun, Corinna Cortes and Christopher Burges for evaluating machine learning models on the handwritten digit classification problem.

Images of digits were taken from a variety of scanned documents, normalized in size and centered. This makes it an excellent dataset for evaluating models, allowing the developer to focus on the machine learning with very little data cleaning or preparation required.

Each image is a 28 by 28 pixel square (784 pixels total). A standard spit of the dataset is used to evaluate and compare models, where 60,000 images are used to train a model and a separate set of 10,000 images are used to test it.

It is a digit recognition task. As such there are 10 digits (0 to 9) or 10 classes to predict. Results are reported using prediction error, which is nothing more than the inverted classification accuracy.

For our experiments, we'll fit various types of neural nets (vanilla multilayered perceptron, 1D CNN & 2D CNN) and evaluate the accuarcies for each of these models on the test set. A diagramatic representation of the models, along with the accuracy achieved by each model is noted below:

Multilayer Perceptron - Accuracy = 98.15%


![Alt text](https://github.com/nildip/DeepLearn_NUMBERS/blob/master/model_images/model_simple_1Dcnn.png?raw=true)


Multilayer 1D CNN - Accuracy = 98.15%


![Alt text](https://github.com/nildip/DeepLearn_NUMBERS/blob/master/model_images/model_simple_1Dcnn.png)


Multilayer 2D CNN - Accuracy = 98.15%


![Alt text](https://github.com/nildip/DeepLearn_NUMBERS/blob/master/model_images/model_simple_1Dcnn.png)


Multilayer Stacked 1D CNN - Accuracy = 98.15%


![Alt text](https://github.com/nildip/DeepLearn_NUMBERS/blob/master/model_images/model_simple_1Dcnn.png)


Multilayer Stacked 2D CNN - Accuracy = 98.15%


![Alt text](https://github.com/nildip/DeepLearn_NUMBERS/blob/master/model_images/model_simple_1Dcnn.png)


Multilayer Multitowered 2D CNN - Accuracy = 98.15%


![Alt text](https://github.com/nildip/DeepLearn_NUMBERS/blob/master/model_images/model_simple_1Dcnn.png)
