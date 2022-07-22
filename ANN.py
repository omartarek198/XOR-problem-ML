
import numpy as np
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))
def dSigmoid (x):
  return x * (1 - x)


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

        self.derivative = dSigmoid


    def train(self, inputs, targets):

        inputs = np.array(inputs)
        targets = np.array(targets)



        hidden_inputs = np.dot((self.weights_input_to_hidden), inputs)


        hidden_outputs = self.activation_function(hidden_inputs)


        final_inputs = np.dot((self.weights_hidden_to_output), hidden_outputs)

        final_outputs = self.activation_function(final_inputs)


        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)


        #
        # var
        gradient_output = self.derivative (final_outputs)

        # // Weight
        # by
        # errors and learing
        # rate
        gradient_output = np.dot(gradient_output,output_errors )
        gradient_output =np.multiply(self.lr,gradient_output)

        #
        # // Gradients
        # for next layer, more back propogation!
        # var
        gradient_hidden = self.derivative(hidden_outputs)

        # // Weight
        # by
        # errors and learning
        # rate
        gradient_hidden = np.dot(gradient_hidden.T,hidden_errors)
        gradient_hidden= np.multiply(self.lr , gradient_hidden)
        #
        # // Change in weights
        # from HIDDEN
        # --> OUTPUT
        # var
        hidden_outputs_T = hidden_outputs.T
        # var
        deltaW_output = np.dot(gradient_output, hidden_outputs_T)
        self.weights_hidden_to_output = np.add(deltaW_output,self.weights_hidden_to_output)

        #
        # // Change in weights
        # from INPUT
        # --> HIDDEN
        # var
        inputs_T = inputs.T

        deltaW_hidden = np.dot(gradient_hidden, inputs_T);
        self.weights_input_to_hidden = np.add(self.weights_input_to_hidden,deltaW_hidden)







    def Guess(self,inputs):
        inputs = np.array(inputs)

        hidden_inputs = np.dot((self.weights_input_to_hidden), inputs)
        hidden_outputs = self.activation_function(hidden_inputs)


        final_inputs = np.dot((self.weights_hidden_to_output), hidden_outputs)
        final_outputs = final_inputs

        return self.activation_function(final_outputs)

