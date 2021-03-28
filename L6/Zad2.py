import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_der(x):
    return x * (1.0 - x)


def relu(x):
    return x * (x > 0)


def relu_der(x):
    return 1 * (x > 0)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_der(x):
    return 1.0 - tanh(x) ** 2


def mse(x, y):
    se = 0
    for i in zip(x, y):
        se += (i[0] - i[1]) ** 2
    return se / len(x)


class NeuralNetwork:
    def __init__(self, x, y, eta):
        self.input = x
        self.weights1 = np.random.rand(10, self.input.shape[1])
        self.weights2 = np.random.rand(1, 10)
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

    def feedforward_tanh(self):
        self.layer1 = tanh(np.dot(self.input, self.weights1.T))
        self.output = tanh(np.dot(self.layer1, self.weights2.T))

    def backprop_tanh(self):
        delta2 = (self.y - self.output) * tanh_der(self.output)
        d_weights2 = self.eta * np.dot(delta2.T, self.layer1)
        delta1 = tanh_der(self.layer1) * np.dot(delta2, self.weights2)
        d_weights1 = self.eta * np.dot(delta1.T, self.input)
        self.weights1 += d_weights1
        self.weights2 += d_weights2


# Jako funkcja aktywacji najepiej sprawdzil sie tanh

time = 2  # czas wyswietlania wykresu podczas pokazu

# parabola
n = 101
r = 50
X = np.linspace(start=-r, stop=r, num=26)
y = X ** 2
plt.scatter(X, y)
plt.title("Zamkniecie okna rozpocznie automatyczna animacje")
plt.suptitle("f(x) = x^2")
plt.grid(True, color='black')
plt.show()
X = np.linspace(start=-r, stop=r, num=n)
X = X / r  # przeskalowanie
y = X ** 2
z = X.reshape(n, 1)
arr = [[1] for i in range(n)]
z = np.hstack((z, np.array(arr)))
y_train = y.reshape(n, 1)
np.random.seed(17)
nn = NeuralNetwork(z, y_train, 0.001)
for step in range(8000):
    nn.feedforward_tanh()
    nn.backprop_tanh()
    if step % 200 == 0:
        plt.scatter(X * r, nn.output * r ** 2)
        print("step: {}, mse = {}".format(step, mse(y, nn.output) * r ** 2))
        plt.title("step: {}, mse = {}".format(step, mse(y, nn.output) * r ** 2))
        plt.grid(True, color='black')
        plt.show(block=False)
        plt.pause(time)
        plt.close()
plt.scatter(X * r, nn.output * r ** 2)
plt.title("Postac ostateczna, mse: {} ".format(mse(y, nn.output) * r ** 2))
plt.grid(True, color='black')
plt.show()

# sinus
n = 161
r = 2
X = np.linspace(start=0, stop=r, num=21)
y = np.sin((3 * np.pi / 2) * X)
plt.scatter(X, y)
plt.title("Zamkniecie okna rozpocznie automatyczna animacje")
plt.suptitle("f(x) = sin(3pi/2 * x)")
plt.grid(True, color='black')
plt.show()
X = np.linspace(start=0, stop=r, num=n)
y = np.sin((3 * np.pi / 2) * X)
X = X / r  # przeskalowanie
z = X.reshape(n, 1)
arr = [[1] for i in range(n)]
z = np.hstack((z, np.array(arr)))
y_train = y.reshape(n, 1)
np.random.seed(17)
nn = NeuralNetwork(z, y_train, 0.001)
for step in range(75000):
    nn.feedforward_tanh()
    nn.backprop_tanh()
    if step % 5000 == 0:
        plt.scatter(X * r, nn.output)
        print("step: {}, mse = {}".format(step, mse(y, nn.output)))
        plt.title("step: {}, mse = {}".format(step, mse(y, nn.output)))
        plt.grid(True, color='black')
        plt.show(block=False)
        plt.pause(time)
        plt.close()
plt.scatter(X * r, nn.output)
plt.title("Postac ostateczna, mse:{}".format(mse(y, nn.output)))
plt.grid(True, color='black')
plt.show()
