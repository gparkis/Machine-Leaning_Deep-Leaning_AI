#single layer feed foward network - data in one direction
#perceptron - this single layer feed forward with weights on input
#given training examples of inputs and proper output - it should learn a function by increasing or decreasing the weights

from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3,1,)) - 1
    def __sigmoid(self, x):
        return 1/(1+exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, trainig_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            output = self.predict(training_set_inputs)
            error = training_set_outputs - output
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

    def predict(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == '__main__':

    neural_network = NeuralNetwork()

    print('Random starting synaptic weights:')
    print(neural_network.synaptic_weights)
    print("")

    training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print('new synaptic weights after training: ')
    print(neural_network.synaptic_weights)
    print("")

    print('considering new situation [1,0,0]: ')
    print(neural_network.predict(array([1,0,0])))
    print("")

