
#from tqdm import trange
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from NueralNetDense import Dense,ReLu,softmax_crossentropy_with_logits,grad_softmax_crossentropy_with_logits
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np



def LabelEncode(dataset, col):
    Label = LabelEncoder()
    Label.fit(dataset[col])
    return Label.transform(dataset[col])
    
def loaddata(data):
    data['prognosis'] = LabelEncode(data,'prognosis')
    y = data['prognosis']
    data = data.drop('prognosis',axis=1)
    return y, data

url = './Dataset/Training.csv'

data = pd.read_csv(url)

train, test = train_test_split(data,test_size=0.3)


y_train,X_train = loaddata(train) 
train = preprocessing.scale(train)
y_val, X_val = loaddata(test)
test = preprocessing.scale(test)

network = []
network.append(Dense(X_train.shape[1],264))
network.append(ReLu())
network.append(Dense(264,512))
network.append(ReLu())
network.append(Dense(512,len(set(y_train))))


def forward(network, X):
    # Compute activations of all network layers by applying them sequentially.
    # Return a list of activations for each layer. 
    
    activations = []
    input = X
    # Looping through each layer
    for l in network:
        activations.append(l.forward(input))
        # Updating input to last layer output
        input = activations[-1]
    
    assert len(activations) == len(network)
    return activations


def predict(network,X):
    # Compute network predictions. Returning indices of largest Logit probability
    
    logits = forward(network,X)[-1]
    return logits.argmax(axis=-1)


def train(network,X,y):
    # Train our network on a given batch of X and y.
    # We first need to run forward to get all layer activations.
    # Then we can run layer.backward going from last to first layer.
    # After we have called backward for all layers, all Dense layers have already made one gradient step.
    
    
    # Get the layer activations
    layer_activations = forward(network,X)
    layer_inputs = [X]+layer_activations  #layer_input[i] is an input for network[i]
    logits = layer_activations[-1]
    
    # Compute the loss and the initial gradient
    loss = softmax_crossentropy_with_logits(logits,y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits,y)
    
    # Propagate gradients through the network
    # Reverse propogation as this is backprop
    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]
        
        loss_grad = layer.backward(layer_inputs[layer_index],loss_grad) #grad w.r.t. input, also weight updates
        
    return np.mean(loss)

'''
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = list(indices[start_idx:start_idx + batchsize])
        yield inputs[excerpt], targets[excerpt]
'''     
        
        
train_log = []
val_log = []

for epoch in range(100):
    train(network,X_train,y_train)
    train_log.append(np.mean(predict(network,X_train)==y_train)*100)
    clear_output()
    print("Epoch",epoch)
    print("Train accuracy:",train_log[-1])
    
print('The accuracy of the test phase is ',np.mean(predict(network,X_val)==y_val))
#
#print('The testing data is ')
#
#print(X_val)

print('The output data is ')

print(y_val)

plt.plot(train_log,label='train accuracy')
plt.plot(val_log,label='val accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()













