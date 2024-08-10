
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
        self.output= np.dot( inputs,self.weights)+ self.biases
class Activation_Relu:
    def forward(self, inputs):
        self.output=np.maximum(0, inputs)
class Activation_Softmax:
    def forward(self,inputs):
        exp_values=np.exp(inputs-np.max(inputs, axis=0, keepdims=True))
        probabilities=exp_values/np.sum(exp_values, axis=0, keepdims=True)
        self.output=probabilities

layer1 =Layer_Dense(2,3)
activation1=Activation_Relu()

layer2= Layer_Dense(3,3)

activation2=Activation_Softmax()

layer1.forward(X)
#print(layer.output)
activation1.forward(layer1.output)
layer2.forward(layer1.output)
activation2.forward(layer2.output)

print(activation2.output[:5])



 


