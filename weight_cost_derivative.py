import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


W1 = np.array([[0.5, -0.2], [0.3, 0.8]])

b1 = np.array([[0.1], [-0.1]])

W2 = np.array([[0.7, -0.5]])

b2 = np.array([[0.2]])

X = np.array([[1], [2]])

Z1 = np.dot(W1, X) + b1

A1 = sigmoid(Z1)

Z2 = np.dot(W2, A1) + b2

A2 = sigmoid(Z2)


Y = np.array([[1]])

C = 0.5 * (A2 - Y) ** 2

print("A2:", A2)
print("Y:", Y)
print("Cost:", C)

# Backpropagation
# Calculate gradients
dC_dA2 = A2 - Y  # Derivative of cost with respect to A2
dA2_dZ2 = sigmoid_derivative(Z2)  # Derivative of A2 with respect to Z2
dC_dZ2 = dC_dA2 * dA2_dZ2  # Chain rule

# Gradients for W2 and b2
dC_dW2 = np.dot(dC_dZ2, A1.T)  # Gradient for W2
dC_db2 = dC_dZ2  # Gradient for b2

# Backpropagation to hidden layer
dZ2_dA1 = W2.T  # Derivative of Z2 with respect to A1
dC_dA1 = np.dot(dZ2_dA1, dC_dZ2)  # Gradient for A1
dA1_dZ1 = sigmoid_derivative(Z1)  # Derivative of A1 with respect to Z1
dC_dZ1 = dC_dA1 * dA1_dZ1  # Chain rule

# Gradients for W1 and b1
dC_dW1 = np.dot(dC_dZ1, X.T)  # Gradient for W1
dC_db1 = dC_dZ1  # Gradient for b1

# Update weights and biases
learning_rate = 0.01
W2 -= learning_rate * dC_dW2
b2 -= learning_rate * dC_db2
W1 -= learning_rate * dC_dW1
b1 -= learning_rate * dC_db1

print("Updated W1:", W1)
print("Updated b1:", b1)
print("Updated W2:", W2)
print("Updated b2:", b2)
