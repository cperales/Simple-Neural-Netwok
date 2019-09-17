import numpy as np
from activation import fun_dict


class ArtificialNeuralNetwork:
    """
    Simple Neural Network with back propagation.
    """
    __name__: str = 'Artificial Neural Network'
    n: int  # Number of elements (instances)
    dim: int  # Dimension of the data (features)
    t: int  # Dimension of the target (labels)
    max_iter: int  # Number of iterations
    neurons: int  # Number of neurons in the input layer
    learning_rate: float  # Step
    # Matrix
    input_weight: np.array
    output_weight: np.array
    bias_input_layer: np.array
    bias_output_layer: np.array
    temp_h: np.array  # Value of the hidden layer before applying activation.
    temp_o: np.array  # Value of the output layer before applying activation.
    # Neuronal functions
    activation = None
    activation_der = None

    def __init__(self):
        """
        Fix the random number generator.
        """
        np.random.seed(0)

    def get_layers(self):
        """
        Feed forward random assignation of the two layers.
        """
        self.get_input_layer()
        self.get_output_layer()

    def get_input_layer(self):
        """
        Weights and bias for the hidden layer.
        """
        self.input_weight = np.random.random((self.dim,
                                              self.neurons)) * 2.0 - 1.0
        # self.bias_input_layer = np.random.random((self.neurons, 1))
        self.bias_input_layer = np.zeros((self.neurons, 1))

    def get_output_layer(self):
        """
        Weight and bias for the output layer.
        """
        self.output_weight = np.random.random((self.neurons, 1)) * 2.0 - 1.0
        # self.bias_output_layer = np.random.random((self.t, 1))
        self.bias_output_layer = np.zeros((self.t, 1))

    def train(self, x, y,
              max_iter: int = 1000,
              neurons: int = 10,
              learning_rate: float = 1.0,
              neuronal_fun='sigmoid'):
        """
        Train the neural network with gradient descent.

        :param x: numpy.array with data (intances and features).
        :param y: numpy.array with the target to predict.
        :param int max_iter: number of iterations for training.
        :param int neurons: number of neurons in the hidden layer.
        :param float learning_rate: step to add in each iteration.
        :param str neuronal_fun: function for activation functions in neurons.
        """
        self.dim = x.shape[1]
        self.t = y.shape[1]
        self.max_iter = max_iter
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.activation = fun_dict[neuronal_fun]['activation']
        self.activation_der = fun_dict[neuronal_fun]['derivative']
        self.get_layers()

        for iteration in range(self.max_iter):
            # print('Iteration =', iteration)
            self.backward(x, y)

    def backward(self, x, y):
        """
        Adjust the weight after the prediction.

        :param x:
        :param y:
        :return:
        """
        hidden_layer, output_layer = self.forward(x)
        error = output_layer - y
        # print('Error =', np.linalg.norm(error))

        # Adjust output layer
        output_delta = error * self.activation_der(self.temp_o)
        # print('Norm of gradient output layer =', np.linalg.norm(output_delta))
        self.bias_output_layer -= np.mean(self.learning_rate * output_delta)
        self.output_weight -= self.learning_rate * np.dot(hidden_layer.T, output_delta)

        # Adjust hidden layer
        hidden_delta = np.dot(output_delta,
                              self.output_weight.T) * self.activation_der(self.temp_h)
        # print('Norm of gradient hidden layer =', np.linalg.norm(hidden_delta))
        self.bias_input_layer -= np.mean(self.learning_rate * hidden_delta)
        self.input_weight -= self.learning_rate * np.dot(x.T, hidden_delta)

    def forward(self, x):
        """
        Calculate the layers in the net when x is added.

        :param x:
        :return: both hidden and output layers.
        """
        self.temp_h = np.dot(x, self.input_weight) + self.bias_input_layer.T
        hidden_layer = self.activation(self.temp_h)
        self.temp_o = np.dot(hidden_layer, self.output_weight) + self.bias_output_layer.T
        output_layer = self.activation(self.temp_o)
        return hidden_layer, output_layer

    def predict(self, x):
        """
        Predict value.

        :param x: Data to predict
        :return:
        """
        _, output = self.forward(x)
        return output


if __name__ == "__main__":
    # Train the neural network using a training set
    train_x = np.array([[0, 0, 1],
                        [1, 1, 1],
                        [1, 0, 1],
                        [0, 1, 1],
                        [1, 1, 1],
                        [0, 0, 0]])
    train_y = np.array([[0],
                        [1],
                        [1],
                        [0],
                        [1],
                        [0]])

    neural_network = ArtificialNeuralNetwork()
    neural_network.train(x=train_x,
                         y=train_y,
                         neuronal_fun='tanh')

    # Test the neural network with a new situation.
    print("Testing data [1, 0, 0] = 1: ")
    test_x = np.array([1, 0, 0])
    print(neural_network.predict(test_x))
