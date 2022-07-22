
import numpy as np
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                        (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5,
                                                         (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate


        self.activation_function = sigmoid

    def train(self, inputs_list, targets_list):

        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T



        hidden_inputs = np.dot((self.weights_input_to_hidden), inputs)


        hidden_outputs = self.activation_function(hidden_inputs)


        final_inputs = np.dot((self.weights_hidden_to_output), hidden_outputs)
        final_outputs = final_inputs


        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)
        hidden_grad = hidden_errors * ((hidden_outputs) * (1 - hidden_outputs))

        self.weights_hidden_to_output += (self.lr) * (hidden_outputs.T * output_errors)
        self.weights_input_to_hidden += (self.lr) * ((inputs.T * hidden_grad))
    def Guess(self,inputs):
        inputs = np.array(inputs, ndmin=2).T

        hidden_inputs = np.dot((self.weights_input_to_hidden), inputs)
        hidden_outputs = self.activation_function(hidden_inputs)


        final_inputs = np.dot((self.weights_hidden_to_output), hidden_outputs)
        final_outputs = final_inputs

        return min(final_outputs,1)

