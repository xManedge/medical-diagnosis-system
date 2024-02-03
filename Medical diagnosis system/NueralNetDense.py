

import numpy as np



def softmax_crossentropy_with_logits(logits,reference_answers):
    logits_for_answers = logits[np.arange(len(logits)),reference_answers]
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
    return xentropy
    
def grad_softmax_crossentropy_with_logits(logits,reference_answers):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)),reference_answers] = 1
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    return (- ones_for_answers + softmax) / logits.shape[0]



class Layer:
    def __init__(self):
        pass


    def forward(self,input):
        return input

    def backward(self, input, grad_output):
        
        size = input.shape[1]
        d_layer_d_input = np.eye(size)
        val = np.dot(grad_output,d_layer_d_input)        
        return val
    

class ReLu(Layer):
    def __init__(self):
        pass
    def forward(self,input):
        return np.maximum(0,input)
    
    def backward(self,input,grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad
    
    
class Dense(Layer):
    def __init__(self,input_units,output_units,learning_rate=0.05):
        self.lr = learning_rate
        self.weights = np.random.normal(loc=0.0,scale = np.sqrt(2/(input_units+output_units)),size = (input_units,output_units))
        print(self.weights.shape)
        self.biases = np.zeros(output_units)
        
    def forward(self,input):
        val = np.dot(input,self.weights) + self.biases
        return val
    
    
    def backward(self,input,grad_output):
        grad_input = np.dot(grad_output,self.weights.T)
        grad_weights = np.dot(input.T,grad_output)
        #weught updation
        grad_biases = grad_output.mean(axis=0)*input.shape[0]
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        self.weights -= self.lr * grad_weights
        self.biases -= self.lr * grad_biases
    
        
        return grad_input
    
    
    
    
    
    
        
    