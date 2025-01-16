import numpy as np

# 1. create network architecture
L = 3
n = [2, 3, 3, 1]

# 2. create weights and biases
W1 = np.random.randn(n[1], n[0])
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)


# 3. create training data and labels
def prepare_data():
    X = np.array(
        [
            [150, 70],
            [254, 73],
            [312, 68],
            [120, 60],
            [154, 61],
            [212, 65],
            [216, 67],
            [145, 67],
            [184, 64],
            [130, 69],
        ]
    )
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    m = 10
    A0 = X.T
    Y = y.reshape(n[L], m)

    return A0, Y


# 4. create activation function
def sigmoid(arr):
    return 1 / (1 + np.exp(-1 * arr))


# 5. create feed forward process
def feed_forward(A0):

    # layer 1 calculations
    Z1 = W1 @ A0 + b1
    A1 = sigmoid(Z1)

    # layer 2 calculations
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    # layer 3 calculations
    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    y_hat = A3
    return y_hat


def cost(y_hat, y):
    """
    y_hat should be a n^L x m matrix
    y should be a n^L x m matrix
    """
    # 1. losses is a n^L x m
    losses = -((y * np.log(y_hat)) + (1 - y) * np.log(1 - y_hat))

    m = y_hat.reshape(-1).shape[0]

    # 2. summing across axis = 1 means we sum across rows,
    #   making this a n^L x 1 matrix
    summed_losses = (1 / m) * np.sum(losses, axis=1)

    # 3. unnecessary, but useful if working with more than one node
    #   in output layer
    return np.sum(summed_losses)


def sigmoid_derivative(A):
    """Derivative of sigmoid function"""
    return A * (1 - A)


def backward_propagation(A0, Y, learning_rate=0.1):
    """
    Compute gradients and update weights/biases
    """
    global W1, W2, W3, b1, b2, b3
    m = A0.shape[1]  # number of training examples

    # Forward pass (saving intermediate values)
    Z1 = W1 @ A0 + b1
    A1 = sigmoid(Z1)

    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    # Backward pass
    # Layer 3 derivatives
    dZ3 = A3 - Y
    dW3 = (1 / m) * (dZ3 @ A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    # Layer 2 derivatives
    dZ2 = (W3.T @ dZ3) * sigmoid_derivative(A2)
    dW2 = (1 / m) * (dZ2 @ A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # Layer 1 derivatives
    dZ1 = (W2.T @ dZ2) * sigmoid_derivative(A1)
    dW1 = (1 / m) * (dZ1 @ A0.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    # Update weights and biases
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1


# Training loop
def train(epochs=1000):
    """Train the neural network"""
    A0, Y = prepare_data()
    costs = []

    for i in range(epochs):
        # Forward propagation
        y_hat = feed_forward(A0)

        # Compute cost
        cost_value = cost(y_hat, Y)

        # Backward propagation
        backward_propagation(A0, Y)

        # Store cost every 100 epochs
        if i % 100 == 0:
            costs.append(cost_value)
            print(f"Epoch {i}, Cost: {cost_value}")

    return costs


# Train the network
costs = train()
