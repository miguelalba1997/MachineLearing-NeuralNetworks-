#Author: Miguel Alba
#Purpose: Implement Python's Class features to create a neural network
#         with basic a Gradient Descent algorithm and forward input feed.

import numpy as np
import random

class NeuralNetwork():
    
    def __init__(self, num_inputs, num_nodes, num_layers, num_outputs, input_list, expected):
        self.learning_rate = .7
        self.num_inputs = num_inputs
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.input_list = input_list
        self.expected = expected
        self.init_wieghts()
        self.init_Neuron() 
        #self.feedforward()
        
    def init_wieghts(self):#this works
        self.wieghts = NeuronLayer.create_wieght_matrix(self, self.num_layers, self.num_nodes, self.num_inputs)

    def init_Neuron(self):
        self.neurons = NeuronLayer.create_neuron(self, self.num_layers, self.num_nodes, self.input_list)
        #print(1, self.neurons[0])
    def feedforward(self):
        self.outputlist = []
        #print(2, np.dot(self.neurons[0], self.wieghts[len(self.wieghts) - 1]))
        
        result = 1/(1 + np.exp(-1 * np.dot(self.neurons[len(self.neurons) - 1],  self.wieghts[len(self.wieghts) - 1])))
        self.outputlist.append(result)
        
    def train(self):#this works
        self.feedforward()

        self.hidden_error = []
        self.error = []

        #calculate the error in our outputs
        for i in range(len(self.outputlist)):            
            self.error.append(self.expected[i] - self.outputlist[i])
        self.error = np.array(self.error)
        
        #backpropogate our error
        for j in range(len(self.wieghts) - 1, 0, -1):           
            if j == (len(self.wieghts) - 1):              
                self.hidden_error.append(np.dot(self.error, self.wieghts[j].T))
            else:
                self.hidden_error.append(np.dot(self.wieghts[j].T, self.hidden_error[j-1]))
        
        #this is the input that goes into our sigmoid activation function
        input_w_ih = np.dot(self.input_list, self.wieghts[0])

        input_w_hh = []
        for k in range(0, len(self.neurons)-1):
            input_w_hh.append(np.dot(self.neurons[k], self.wieghts[k]))
        input_w_hh = np.array(input_w_hh)
        
        input_w_ho = np.dot(self.neurons[len(self.neurons)-1], self.wieghts[1])

        #get the changes we'll need to our wieght tensor
        self.delta_w_ih = (self.learning_rate * np.outer(np.reshape(self.input_list, (2,1)),                                                                                    (self.hidden_error[0] * Neuron.sigmoid_derivative(self, input_w_ih))))
        
        self.delta_w_hh = []
        for l in range(len(input_w_hh)):
            self.delta_w_hh.append(self.learning_rate * 
                np.dot((self.hidden_error * Neuron.sigmoid_derivative(self, input_w_hh[l])).reshape((-1,1)), self.neurons[l+1]))
        self.delta_w_hh = np.array(self.delta_w_hh)

        self.delta_w_ho = (self.learning_rate * np.outer(np.reshape(self.neurons[len(self.neurons) - 1],(3,1)), 
                            (self.error * Neuron.sigmoid_derivative(self, input_w_ho))))

        #print("wieght matrix:\n", self.wieghts, '\ndelta ih:\n', self.delta_w_ih, '\ndelta ho:\n', self.delta_w_ho)
        
        for w in range(len(self.wieghts)):
            for u in range(len(self.wieghts[w])):
                for v in range(len(self.wieghts[w][u])):
                    if w == 0:
                        self.wieghts[w][u][v] += self.delta_w_ih[u][v]
                                 
                    elif w == len(self.wieghts) - 1:
                        self.wieghts[w][u][v] += self.delta_w_ho[u][v]
                                        
                    else:
                        self.wieghts[w][u][v] += self.delta_w_hh[u][v]
                        
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
            if i == 0:
                for j in range(num_inputs):
                    self.wieghtlist[i].append([])
                    for k in range(num_nodes):
                        self.wieghtlist[i][j].append(random.random())
                    self.wieghtlist[i][j] = np.array(self.wieghtlist[i][j])
            elif i == layers:
                for j in range(num_nodes):
                    self.wieghtlist[i].append([])
                    for k in range(self.num_outputs):
                        self.wieghtlist[i][j].append(random.random())
                    self.wieghtlist[i][j] = np.array(self.wieghtlist[i][j])
            else:
                for j in range(num_nodes):                       
                    self.wieghtlist[i].append([])
                    for k in range(num_nodes):
                        self.wieghtlist[i][j].append(random.random())
                    self.wieghtlist[i][j] = np.array(self.wieghtlist[i][j])
            self.wieghtlist[i] = np.array(self.wieghtlist[i])
        self.wieghtlist = np.array(self.wieghtlist)
        return self.wieghtlist

    def create_neuron(self, layers, num_nodes, input_list):
        random.seed(1)
        self.neuronlist = []        
        for i in range(layers):
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
            output = 1/(1 + np.exp(-1 * x[i]))#sigmoid func
            self.output.append(output)                    
        return self.output
    
    def sigmoid_derivative(self, x):
        return (np.exp(-1 * x))/(1 + np.exp(-1 * x))**2

nn = NeuralNetwork(2 ,3, 1, 2, input_list = [1,1], expected = [1,0])
"""
print("input list:", nn.input_list, "\n")
print("wieght list:\n", len(nn.wieghts[0]),len(nn.wieghts[0][0]), "\n")
print("neuron list:",nn.neurons, "\n")
"""
for i in range(0, 300):
    nn.train()
    
    print(i, nn.error)
    """
    print("output:",nn.outputlist,"\n")
    print("hidden error list:", nn.hidden_error, "\n")
    print("delta input to hidden:\n",nn.delta_w_ih)
    print("delta hidden to hidden:\n",nn.delta_w_hh)       
    print("delta hidden to output:\n",nn.delta_w_ho)
    """
