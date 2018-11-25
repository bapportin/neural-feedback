import numpy as np
import context
import layers
import random

def print_stat(epoch,loss,err):
    print "epoch: ",epoch,"loss: ",loss,"err: ",err

class BaseModel(layers.Layer):
    def mini_batches(self,X,y,batch_size):
        batches=[]
        for i in xrange(((len(X)-1)/batch_size)+1):
            batches.append([])
        for i,x in enumerate(X):
            ib=(i%len(batches))
            if ib==0:
                random.shuffle(batches)
            batches[ib].append((x,y[i]))
        return batches
    def fit(self,X,y,batch_size=10,epochs=1,loss=lambda x: np.sum(x),lr=0.1,finish=lambda x: x<0.1,print_stat=print_stat):
        for epoch in xrange(epochs):
            batches=self.mini_batches(X,y,batch_size)
            ex=None
            for data in batches:
                #print data
                tmp=[]
                for i,(bx,by) in enumerate(data):
                    #print (i,bx,by)
                    ctx=context.Context()
                    r=self.forward(bx,str(id(self)),ctx)
                    e=by-r
                    #print ("e",e)
                    tmp.append((e,ctx))
                for e,ctx in tmp:
                    #print e,ctx,data
                    self.backward(e,lr*(1.0/len(tmp)),str(id(self)),ctx)
                    if ex is None:
                        ex=np.abs(e)
                    else:
                        ex+=np.abs(e)            
            ex=ex/len(X)
            lo=loss(ex)
            print_stat(epoch,lo,ex)
            if finish(lo):
                break
            
    def predict(self,X):
        return self.forward(X)
        
        


class Sequential(BaseModel):
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

