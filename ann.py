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
    input_weights: np.array
    output_weights: np.array
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
        self.input_weights = np.random.random((self.dim,
                                               self.neurons)) * 2.0 - 1.0
        self.bias_input_layer = np.random.random((self.neurons, 1))
        # self.bias_input_layer = np.zeros((self.neurons, 1))  # Without bias

    def get_output_layer(self):
        self.output_weights = np.random.random((self.neurons, 1)) * 2.0 - 1.0
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
            output = self.predict(train_data)
            self.adjust(train_target, output)

    def adjust(self, train_target, output):
        """
        Adjust the weights after the prediction.

        :param train_target:
        :param output:
        :return:
        """
        error = train_target - output
        output_delta = error * sigmoid_derivative(output)

        adj_output = np.dot(self.layer_1.T, output_delta)
        self.output_weights += self.learning_rate * adj_output

        hidden_delta = np.dot(output_delta, self.output_weights.T) * sigmoid_derivative(self.layer_1)
        adj_hidden = np.dot(train_data.T, hidden_delta)

        self.input_weights += self.learning_rate * adj_hidden

    # The neural network get_indicators.
    def predict(self, test_data):
        """
        Predict value.

        :param test_data:
        :return:
        """
        z_1 = np.dot(test_data, self.input_weights) + self.bias_input_layer.T
        self.layer_1 = sigmoid(z_1)
        z_2 = np.dot(self.layer_1, self.output_weights) + self.bias_output_layer.T
        output = sigmoid(z_2)
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
