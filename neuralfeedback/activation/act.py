import numpy as np

class Activation:
    def forward(self,x):
        raise NotImplementedError
    def gradient(self,x):
        raise NotImplementedError

class Tanh(Activation):
    def forward(self,x):
        return np.tanh(x)
    def gradient(self,x):
        return 1-np.power(x,2)

class Sigmoid(Activation):
    def __init__(self,yscale=1.0):
        self.yscale=yscale
    def forward(self,x):
        return self.yscale / (1.0 + np.exp(-x))
    def gradient(self,x):
        #ret=np.ones((2,)+x.shape)*0.1
        #ret[0]=(x * (1 - x))
        #return np.max(ret,axis=0)
        x=x/self.yscale
        return (x * (1 - x))
    
class Softmax(Activation):
    def forward(self, x):
        shiftx=x-np.max(x)
        exp = np.exp(shift)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def gradient(self, x):
        return x * (1 - x)

class Relu(Activation):
    def forward(self, x):
        return np.maximum(x, 0, x)

    def gradient(self, x):
        return 1*(x > 0)

class LeakyRelu(Activation):
    def forward(self, x):
        return np.maximum(x, 0.01 * x, x)

    def gradient(self, x):
        return 0.01 + 0.99 * (x > 0)
    
    
class Linear(Activation):
    def forward(self, x):
        return x

    def gradient(self, x):
        return np.ones_like(x)


class Logabs(Activation):
    def forward(self, x):
        return np.sign(x)*np.log(np.abs(x)+1)

    def gradient(self, x):
        return np.exp(-np.abs(x))
