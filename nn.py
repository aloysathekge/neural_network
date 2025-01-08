import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
X, y = spiral_data(samples=100, classes=3)


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_loss = self.forward(output, y)

        data_loss = np.mean(sample_loss)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


layer1 = Layer_Dense(2, 3)
activation1 = Activation_Relu()

layer2 = Layer_Dense(3, 3)

activation2 = Activation_Softmax()


layer1.forward(X)
# print(layer.output)
activation1.forward(layer1.output)
layer2.forward(layer1.output)
activation2.forward(layer2.output)
loss_function = Loss_CategoricalCrossentropy()

print(activation2.output[:5])
loss = loss_function.calculate(activation2.output, y)
print("loss", loss)
