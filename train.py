
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pickle
import numpy as np
import random
import os

def save(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

#Generate x_train,y_train,x_test,y_test
def generate(path1,paht2,T_fNum,T_nfNum):
    files1=os.listdir(path1)
    files2=os.listdir(path2)
    
    y=np.ones((T_fNum+T_nfNum,1))
    y[T_fNum:]=-1
    
    idx=random.sample(range(T_fNum+T_nfNum),T_fNum+T_nfNum)
    
    T_fNum=random.sample(range(len(files1)),T_fNum)
    T_nfNum=random.sample(range(len(files2)),T_nfNum)
    
    os.chdir(path1)
    x=load(files1[T_fNum[0]]).reshape((1,-1))
    T_fNum=T_fNum[1:]
    
    for i in T_fNum:
        x=np.r_[x,load(files1[i]).reshape((1,-1))]
        pass
    os.chdir(path2)
    for i in T_nfNum:
        x=np.r_[x,load(files2[i]).reshape((1,-1))]
        pass
    
    x=x[idx]
    y=y[idx]
    return x,y

if __name__ == "__main__":
    # write your code here
    path1="D:/datasets/NPD_face"
    path2="D:/datasets/NPD_nonface"
    
    #Num of Train sample
    Tr_fNum=300
    Tr_nfNum=300
    
    #Num of Test sample
    Te_fNum=150
    Te_nfNum=150
    
    #Truns
    turn=10
    
    #All Classifier
    weak=[]
    a=[]
    
    #Get Samples
    x_train,y_train=generate(path1,path2,Tr_fNum,Tr_nfNum)
    x_test,y_test=generate(path1,path2,Te_fNum,Te_nfNum)
    
    #Initial weight
    w=np.ones((Tr_fNum+Tr_nfNum))/(Tr_fNum+Tr_nfNum)
    
    #Fit Process:
    for i in range(turn):
        each=DecisionTreeClassifier(max_depth=3)
        #Train a Classifier
        each.fit(x_train,y_train,w)
        weak.append(each)
        #Print one Classifer's fault
        #fault=1-each.score(x_train,y_train,w)
        fault=np.dot(w,(y_train.reshape(-1)!=(weak[i].predict(x_train))))
        print("Single Classifier fault=%s"%fault)
        #Get a
        temp_a=1/2*np.log((1-fault)/fault)
        a.append(temp_a)
        #Get normalization term
        z=np.dot(w,np.exp(-temp_a*y_train*each.predict(x_train).reshape(-1,1)))
        #Renew weight
        w=w/z*np.exp(-temp_a*y_train*each.predict(x_train).reshape(-1,1)).reshape(-1)
        print("score=%s"%(each.score(x_test,y_test)))
    
    #Predict results
    pre=0;
    for i in range(turn):
        pre+=a[i]*weak[i].predict(x_test)
    pre=np.sign(pre).reshape(-1)
    y_test=y_test.reshape(-1)
    target_names = ['Class 0','Class 1']
    
    #Save report
    os.chdir("D:/datasets")
    save(classification_report(y_test, pre, target_names=target_names),"report.txt")
    
    #Caculate the accurancy
    pre=((pre!=y_test).sum(0)/len(pre))
    print("Total accurancy=%s"%(1-pre))
    
    pass

