#Author: Miguel Alba
#Purpose: The goal of this program is to create a simple
#         nueral network that has forward and backward 
#         feeding aspects *from scratch*.

import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_wieghts = 2 * np.random.random((3, 1)) - 1

    #this is what is known as an activation function
    #It is a function that gives us a probabilistic distribution
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    #Gradient descent is the concept that the gradient or 
    #derivatives of a functions dimensional components(x,y,z)
    #can tell you what steps you should take to find the relative
    #minima of that function. In Machine learning, you really want
    #to minimize the the loss function.
    def Gradient_Descent(self, x):
       return x * (1 - x)
    
    #now we impelement a forward feeding training system. This
    #is where we *feed* the network data and train it to match
    #the given output
    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.Gradient_Descent(output))
            self.synaptic_wieghts += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_wieghts))
        return output

if __name__ == "__main__":
    #initialize the network
    nn = NeuralNetwork()
    print("Random Synaptic wieghts:\n", nn.synaptic_wieghts)
    #*trian* the network
    training_inputs = np.array([[0,0,1], 
                     [1,1,1], 
                     [1,0,1],
                     [0,1,1]])
    training_outputs  = np.array([[0,1,1,0]]).T
    nn.train(training_inputs, training_outputs, 100000)
    print("Synaptic wieghts after forward feeding:\n", nn.synaptic_wieghts)
    
    #test your network to see how it behaves
    a = str(input("Input 1:"))
    b = str(input("Input 2:"))
    c = str(input("Input 3:"))
      
    Certainty = nn.think(np.array([a,b,c]))
    if Certainty > 0.5:
        result = 1
    else:
        result = 0
    print("Outcome:", result)

    

