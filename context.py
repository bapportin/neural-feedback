class Context:
    def __init__(self,storeData=True):
        self.data={}
        self.storeData=storeData
    def put(self,layer,level,inp,out):
        if self.storeData:
            self.data[(layer,level)]=(inp,out)
    def get(self,layer,level):
        return self.data.get((layer,level))

_dummyCtx=Context(False)    
