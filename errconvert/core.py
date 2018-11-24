import numpy as np

def dupResample(err,out):
        if err.shape[0]<out.shape[0]:
            #shape up by repeat
            x=((out.shape[0]-1)/err.shape[0])+1
            e=np.tile(err,x)[:out.shape[0]]
        elif err.shape[0]>out.shape[0]:
            #shape down by summing up
            e=np.zeros(out.shape[0])
            for i in xrange(0,err.shape[0],e.shape[0]):
                part=err[i:i+e.shape[0]]
                e[:part.shape[0]]+=part
        else:
            #same shape, do noting
            e=err
        return e    

def linearResample(err,out):
    return np.interp(np.linspace(0,len(err),len(out)),np.linspace(0,len(err),len(err)),err)

def linearResampleScale(err,out):
    ret=linearResample(err,out)
    f=float(len(err))/len(out)
    return ret*f
