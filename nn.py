
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
        print( "Weights are \n")
        print( self.weights)
    # Forward pass
    def forward(self, inputs):
        self.output= np.dot( inputs,self.weights)+ self.biases
class Activation_Relu:
    def forward(self, inputs):
        self.output=np.maximum(0, inputs)

layer= Layer_Dense(2,3)
activation1=Activation_Relu()

layer.forward(X)
#print(layer.output)
activation1.forward(layer.output)
print("Activated Neurons are:\n")
print(activation1.output)


 


