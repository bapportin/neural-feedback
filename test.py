import tests
import os
import traceback
import imp
import sys
import re

def loadTestModules(basedir,testdir):
    ret={}
    for root,dirs,files in os.walk(testdir):
        mod=root[len(basedir):].replace("//","/").replace("/",".").replace("\\",".")
        while mod and mod[0]==".":
            mod=mod[1:]
        while mod and mod[-1]==".":
            mod=mod[:-1]
        for file in files:
            if file.endswith(".py") and file!="__init__.py":
                m=mod+"."+(file[:-3])
                try:
                    __import__(m)
                    #ret[m]=sys.modules[m]
                    for name in dir(sys.modules[m]):
                        v=getattr(sys.modules[m],name)
                        if name.startswith("test_") and callable(v):
                            ret[m+"."+name]=v
                except:
                    traceback.print_exc()
    return ret
        

if __name__=="__main__":
    basedir=os.path.dirname(os.path.abspath(__file__))
    testdir=os.path.join(basedir,"tests")
    tests=loadTestModules(basedir,testdir)
    if len(sys.argv)<2:
        print sys.argv[0]+" regexp"
        print "    executes all tests with matching regexp"
        print "tests:"
        for k in sorted(tests.keys()):
            print k
    else:
        for k,v in sorted(tests.items()):
            if re.match(sys.argv[1],k):
                print "executing: "+k
                v()
        
        
    #print TESTMODS
        


