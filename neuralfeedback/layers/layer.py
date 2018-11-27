import numpy as np
import context
import activation
import errconvert

_dummyCtx=context._dummyCtx

class Layer:
    def forward(self,X,level,ctx=_dummyCtx):
        return X

    def backward(self,e,learnrate,level,ctx):
        pass

class FullyConnectedLayer(Layer):
    def __init__(self,insize,outsize,activation=activation.Logabs(),convert=errconvert.linearResample):
        self.activation=activation
        self.insize=insize
        self.outsize=outsize
        self.W=np.random.randn(insize, outsize) * (0.1 / insize)
        #self.W=np.zeros((insize, outsize))# * (1 / np.sqrt(insize))
        self.b=np.zeros(self.outsize)
        self.convert=convert

    def forward(self,X,level=0,ctx=_dummyCtx):
        ret=self.activation.forward(np.dot(X, self.W)+self.b)
        ctx.put(self,level,X,ret)
        return ret

    def backward(self,e,lr,level,ctx):
        inp,out=ctx.get(self,level)
        e=self.convert(e,out)
        e*=self.activation.gradient(out)
        self.W+=np.outer(inp,e)*lr
        self.b+=e*lr

class LocallyConnectedLayer(Layer):
    def __init__(self,insize,outsize,coresize,activation=activation.Logabs(),convert=errconvert.linearResample):
        self.activation=activation
        self.insize=insize
        self.outsize=outsize
        self.coresize=coresize
        #self.W=np.random.randn(insize, outsize) * (1 / np.sqrt(insize))
        self.W=np.zeros((coresize, outsize))
        self.b=np.zeros(self.outsize)
        self.convert=convert

    def forward(self,X,level=0,ctx=_dummyCtx):
        ret=self.activation.forward(np.dot(X, self.W)+self.b)
        ctx.put(self,level,X,ret)
        return ret

    def backward(self,e,lr,level,ctx):
        inp,out=ctx.get(self,level)
        e=self.convert(e,out)
        e*=self.activation.gradient(out)
        self.W+=np.outer(inp,e)*lr
        self.b+=e*lr
