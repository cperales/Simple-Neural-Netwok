import numpy as np


def sigmoid(x):
    """
    Sigmoid function. It can be replaced with scipy.special.expit.

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    """
    Derivate of the sigmoid function.
    We assume y is already sigmoided.

    :param y:
    :return:
    """
    return y * (1.0 - y)


class ArtificialNeuralNetwork(object):
    """
    Simple Neural Network with backpropagation.
    """
    __name__: str = 'Artificial Neural Network'
    n: int  # Number of elements (instances)
    dim: int  # Dimension of the data (features)
    max_iter: int  # Number of iterations
    neurons: int # Number of neurons in the input layer
    learning_rate: float  # Step
    # Matrixes
    input_weight: np.array
    output_weight: np.array
    bias_input_layer: np.array
    bias_output_layer: np.array
    layer_1: np.array  # Value of the output layer, before applying sigmoid.

    def __init__(self):
        """
        Fix the random number generator.
        """
        np.random.seed(1)

    def get_layers(self):
        """
        Feedforward random assignation of the two layers.
        """
        self.get_input_layer()
        self.get_output_layer()

    def get_input_layer(self):
        """
        Weights and bias for input layer.
        """
        self.input_weight = np.random.random((self.dim,
                                               self.neurons)) * 2.0 - 1.0
        self.bias_input_layer = np.random.random((self.neurons, 1))
        # self.bias_input_layer = np.zeros((self.neurons, 1))  # Without bias

    def get_output_layer(self):
        self.output_weight = np.random.random((self.neurons, 1)) * 2.0 - 1.0
        self.bias_output_layer = np.random.random((1, 1))
        # self.bias_output_layer = np.zeros((self.t, 1))  # Without bias

    def train(self, train_data, train_target, max_iter=1000, neurons=10, learning_rate=2.0):
        """
        Train the neural network with gradient descent.

        :param train_data: numpy.array with data (intances and features).
        :param train_target: numpy.array with the target to predict.
        :param int max_iter: number of iterations for training.
        :param int neurons: number of neurons in the hidden layer.
        :param float learning_rate: step to add in each iteration.
        """
        self.dim = train_data.shape[1]
        self.max_iter = max_iter
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.get_layers()

        for iteration in range(self.max_iter):
            self.backward(train_data, train_target)

    def backward(self, train_data, train_target):
        """
        Adjust the weight after the prediction.

        :param train_data:
        :param train_target:
        :return:
        """
        hidden_layer, output = self.forward(train_data)
        error = output - train_target

        # Adjust output layer
        output_delta = error * sigmoid_derivative(output)
        adj_output = np.dot(hidden_layer.T, output_delta)
        self.output_weight -= self.learning_rate * adj_output
        self.bias_output_layer -= (self.learning_rate * output_delta).mean()

        # Adjust hidden layer
        hidden_delta = np.dot(output_delta, self.output_weight.T) * \
                       sigmoid_derivative(hidden_layer)
        adj_hidden = np.dot(train_data.T, hidden_delta)
        self.input_weight -= self.learning_rate * adj_hidden
        self.bias_input_layer -= (self.learning_rate * hidden_delta).mean()

    def forward(self, data):
        """
        Calculate the layers in the net when data is added.

        :param data:
        :return: both hidden and output layers.
        """
        z_1 = np.dot(data, self.input_weight) + self.bias_input_layer.T
        layer_1 = sigmoid(z_1)
        z_2 = np.dot(layer_1, self.output_weight) + self.bias_output_layer.T
        output = sigmoid(z_2)
        return layer_1, output

    def predict(self, test_data):
        """
        Predict value.

        :param test_data:
        :return:
        """
        _, output = self.forward(test_data)
        return output


if __name__ == "__main__":
    # Training set
    train_data = np.array([[0, 0, 1],
                           [1, 1, 1],
                           [1, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1],
                           [0, 0, 0]])
    train_target = np.array([[0],
                             [1],
                             [1],
                             [0],
                             [1],
                             [0]])

    # Train the neural network using a training set.
    neural_network = ArtificialNeuralNetwork()
    neural_network.train(train_data=train_data,
                         train_target=train_target)

    # Test the neural network with a new situation.
    print("Testing data [1, 0, 0] = 1: ")
    test_data = np.array([1, 0, 0])
    print(neural_network.predict(test_data))
