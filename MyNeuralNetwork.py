#Author: Miguel Alba
#Purpose: Implement Python's Class features to create a neural network
#         with basic a Gradient Descent algorithm and forward input feed.

import numpy as np
import random

class NeuralNetwork():
    Learning_rate = .7
    
    def __init__(self, num_inputs, num_nodes, num_layers, num_outputs, input_list, expected):
        self.num_inputs = num_inputs
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.input_list = input_list
        self.expected = expected
        self.init_wieghts()
        self.init_Neuron() 
        self.feedforward()

    def init_wieghts(self):
        self.wieghts = NeuronLayer.create_wieght_matrix(self, self.num_layers, self.num_nodes, self.num_inputs)
    
    def init_Neuron(self):
        self.neurons = NeuronLayer.create_neuron(self, self.num_layers, self.num_nodes, self.input_list)

    def feedforward(self):
        self.outputlist = []
        self.error = []
        for i in range(self.num_nodes):
            result = 1/(1 + np.exp(-1 * self.neurons[len(self.neurons)-1][i]))
            self.outputlist.append(result)
        for j in range(len(self.outputlist)):            
            self.error.append(0.5 * (self.expected[j] - self.outputlist[j])**2)
                
class NeuronLayer():
    def __init__(self, num_inputs, num_nodes, num_outputs, input_list, layers):
        self.num_inputs = num_inputs
        self.num_nodes = num_nodes
        self.num_outputs = num_outputs
        self.input_list = input_list
        self.layers = layers
        self.create_neuron()
        self.create_wieght_matrix()
    
    def create_wieght_matrix(self, layers, num_nodes, num_inputs):
        random.seed(1)
        self.wieghtlist = []
        for i in range(layers + 1):
            self.wieghtlist.append([])
            for j in range(num_nodes):
                self.wieghtlist[i].append([])
                if i == 0:
                    for k in range(num_inputs):
                        self.wieghtlist[i][j].append(random.random())
                else:
                    for k in range(num_nodes):
                        self.wieghtlist[i][j].append(random.random())
                self.wieghtlist[i][j] = np.array(self.wieghtlist[i][j])
        self.wieghtlist = np.array(self.wieghtlist)
        return self.wieghtlist

    def create_neuron(self, layers, num_nodes, input_list):
        random.seed(1)
        self.neuronlist = []        
        for i in range(layers + 1):
            if i == 0:
                node = Neuron(input_list, self.wieghtlist[i])
                self.neuronlist.append(node.output)
            else:
                node = Neuron(self.neuronlist[i-1], self.wieghtlist[i])
                self.neuronlist.append(node.output)
        return self.neuronlist

class Neuron():
    def __init__(self, inputs, wieghts):
        self.inputs = inputs
        self.wieghts = wieghts
        self.output()

    def output(self):
        x = np.dot(self.inputs, self.wieghts)
        self.output = []
        for i in range(len(x)):
            output = 1/(1 + np.exp(-1 * x[i]))
            self.output.append(output)                    
        return self.output
    
    def sigmoid_derivative(self):
        return (np.exp(-1 * x))/(1 + np.exp(-1 * x))**2

nn = NeuralNetwork(2,2,2,2,input_list = [1,1], expected = [1,0])
print("neuron list:",nn.neurons, "\n")
print("output:",nn.outputlist,"\n")        
print("error:",nn.error)
            
