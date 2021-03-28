import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_der(x):
    return x * (1.0 - x)


def relu(x):
    return x * (x > 0)


def relu_der(x):
    return 1 * (x > 0)


class NeuralNetwork:
    def __init__(self, x, y, eta):
        self.input = x
        self.weights1 = np.random.rand(4, self.input.shape[1])
        self.weights2 = np.random.rand(1, 4)
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.eta = eta

    def feedforward_sigmoid(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1.T))
        self.output = sigmoid(np.dot(self.layer1, self.weights2.T))

    def backprop_sigmoid(self):
        delta2 = (self.y - self.output) * sigmoid_der(self.output)
        d_weights2 = self.eta * np.dot(delta2.T, self.layer1)
        delta1 = sigmoid_der(self.layer1) * np.dot(delta2, self.weights2)
        d_weights1 = self.eta * np.dot(delta1.T, self.input)
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def feedforward_relu(self):
        self.layer1 = relu(np.dot(self.input, self.weights1.T))
        self.output = relu(np.dot(self.layer1, self.weights2.T))

    def backprop_relu(self):
        delta2 = (self.y - self.output) * relu_der(self.output)
        d_weights2 = self.eta * np.dot(delta2.T, self.layer1)
        delta1 = relu_der(self.layer1) * np.dot(delta2, self.weights2)
        d_weights1 = self.eta * np.dot(delta1.T, self.input)
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def feedforward_relu_to_sigmoid(self):
        self.layer1 = relu(np.dot(self.input, self.weights1.T))
        self.output = sigmoid(np.dot(self.layer1, self.weights2.T))

    def backprop_relu_to_sigmoid(self):
        delta2 = (self.y - self.output) * sigmoid_der(self.output)
        d_weights2 = self.eta * np.dot(delta2.T, self.layer1)
        delta1 = relu_der(self.layer1) * np.dot(delta2, self.weights2)
        d_weights1 = self.eta * np.dot(delta1.T, self.input)
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def feedforward_sigmoid_to_relu(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1.T))
        self.output = relu(np.dot(self.layer1, self.weights2.T))

    def backprop_sigmoid_to_relu(self):
        delta2 = (self.y - self.output) * relu_der(self.output)
        d_weights2 = self.eta * np.dot(delta2.T, self.layer1)
        delta1 = sigmoid_der(self.layer1) * np.dot(delta2, self.weights2)
        d_weights1 = self.eta * np.dot(delta1.T, self.input)
        self.weights1 += d_weights1
        self.weights2 += d_weights2


np.set_printoptions(precision=3, suppress=True)
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
# wektor wag ma tyle elementow ile wynosi szerokosc macierzy wejscowej,
# kolumna jedynek jest przeznaczona dla wagi wyrazu wolnego, w tym miejscu bedzie budowany bias sluzacy do wywazenia
_xor = np.array([[0],
                 [1],
                 [1],
                 [0]])
_or = np.array([[0],
                [1],
                [1],
                [1]])
_and = np.array([[0],
                 [0],
                 [0],
                 [1]])
print("sigmoid:")
nn = NeuralNetwork(X, _xor, 0.1)
for i in range(10000):
    nn.feedforward_sigmoid()
    nn.backprop_sigmoid()
print("xor:")
print(nn.output)

nn = NeuralNetwork(X, _or, 0.1)
for i in range(10000):
    nn.feedforward_sigmoid()
    nn.backprop_sigmoid()
print("or:")
print(nn.output)

nn = NeuralNetwork(X, _and, 0.1)
for i in range(10000):
    nn.feedforward_sigmoid()
    nn.backprop_sigmoid()
print("and:")
print(nn.output)

print("\nrelu:")
np.random.seed(1)
nn = NeuralNetwork(X, _xor, 0.1)
for i in range(10000):
    nn.feedforward_relu()
    nn.backprop_relu()
print("xor:")
print(nn.output)

np.random.seed(15)
nn = NeuralNetwork(X, _or, 0.1)
for i in range(10000):
    nn.feedforward_relu()
    nn.backprop_relu()
print("or:")
print(nn.output)

np.random.seed(1)
nn = NeuralNetwork(X, _and, 0.1)
for i in range(10000):
    nn.feedforward_relu()
    nn.backprop_relu()
print("and:")
print(nn.output)

print("\nrelu_to_sigmoid:")
np.random.seed(15)
nn = NeuralNetwork(X, _xor, 0.1)
for i in range(10000):
    nn.feedforward_relu_to_sigmoid()
    nn.backprop_relu_to_sigmoid()
print("xor:")
print(nn.output)

np.random.seed(5)
nn = NeuralNetwork(X, _or, 0.1)
for i in range(10000):
    nn.feedforward_relu_to_sigmoid()
    nn.backprop_relu_to_sigmoid()
print("or:")
print(nn.output)

np.random.seed(3)
nn = NeuralNetwork(X, _and, 0.1)
for i in range(10000):
    nn.feedforward_relu_to_sigmoid()
    nn.backprop_relu_to_sigmoid()
print("and:")
print(nn.output)

print("\nsigmoid_to_relu:")
nn = NeuralNetwork(X, _xor, 0.1)
for i in range(10000):
    nn.feedforward_sigmoid_to_relu()
    nn.backprop_sigmoid_to_relu()
print("xor:")
print(nn.output)

nn = NeuralNetwork(X, _or, 0.1)
for i in range(10000):
    nn.feedforward_sigmoid_to_relu()
    nn.backprop_sigmoid_to_relu()
print("or:")
print(nn.output)

nn = NeuralNetwork(X, _and, 0.1)
for i in range(10000):
    nn.feedforward_sigmoid_to_relu()
    nn.backprop_sigmoid_to_relu()
print("and:")
print(nn.output)

# np.random.seed(15) relu OR
# np.random.seed(1) AND
# np.random.seed(1) XOR

# np.random.seed(15) relu_to_sigmoid XOR
# np.random.seed(3) AND
# np.random.seed(5) OR

# sigmoid_to_relu XOR
# OR
