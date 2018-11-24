import numpy as np
import context
import layers


class Sequential(layers.Layer):
    def __init__(self):
        self._layers=[]

    def add(self,l):
        self._layers.append(l)

    def forward(self,X,level=0,ctx=context._dummyCtx):
        ret=X
        level=str(level)
        for i,l in enumerate(self._layers):
            ret=l.forward(ret,level+"."+str(i),ctx)
        return ret

    def backward(self,e,lr,level,ctx):
        level=str(level)
        for i,l in enumerate(self._layers):
            l.backward(e,lr,level+"."+str(i),ctx)

    def fit(self,data,epochs=1,loss=lambda x: np.sum(x),lr=0.7):
        #data=zip(inputs,outputs)
        for i in xrange(epochs):
            ex=None
            for X,y in data:
                ctx=context.Context()
                r=self.forward(X,str(id(self)),ctx)
                e=y-r
                #print r,y
                self.backward(e,lr,str(id(self)),ctx)
                if ex is None:
                    ex=np.abs(e)
                else:
                    ex+=np.abs(e)
            ex=ex/len(data)
            #print "epoch: ",i,"loss: ",loss(ex),"err: ",ex

    def predict(self,X):
        return self.forward(X)

    def fit1(self,data,loss=lambda x: np.sum(x),lr=0.1,finish=lambda x: x<0.1):
        #data=zip(inputs,outputs)
        i=0
        while True:
        #for i in xrange(epochs):
            ex=None
            for X,y in data:
                ctx=context.Context()
                r=self.forward(X,str(id(self)),ctx)
                e=y-r
                print r,y
                self.backward(e,lr,str(id(self)),ctx)
                if ex is None:
                    ex=np.abs(e)
                else:
                    ex+=np.abs(e)
            ex=ex/len(data)
            print "epoch: ",i,"loss: ",loss(ex),"err: ",ex
            if finish(loss(ex)):
                break
            i+=1


class MultiModel(layers.Layer):
    def __init__(self):
        self._data={}

    def add(self,l,name):
        if not name in self.data:
            self.data[name]=Sequential()
        self.data[name].add(l)

    def forward(self,X,level=0,ctx=context._dummyCtx):
        result=[]
        level=str(level)
        for name,model in sorted(self.data.items()):
            result.append(model.forward(X,level+"."+str(name),ctx))
        return np.concatenate(result)

    def backward(self,e,lr,level,ctx):
        level=str(level)
        for name,model in sorted(self.data.items()):
            l.backward(e,lr,level+"."+str(name),ctx)

    def fit(self,data,epochs=1,loss=lambda x: np.sum(x),lr=0.3):
        #data=zip(inputs,outputs)
        for i in xrange(epochs):
            ex=None
            for X,y in data:
                ctx=context.Context()
                r=self.forward(X,str(id(self)),ctx)
                e=y-r
                self.backward(e,lr,str(id(self)),ctx)
                if ex is None:
                    ex=np.abs(e)
                else:
                    ex+=np.abs(e)
            ex=ex/len(data)
            print "epoch: ",i,"loss: ",loss(ex),"err: ",ex

    def predict(self,X):
        return self.forward(X)
