import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt
import get_dataset
import get_testset
from sklearn import svm
import pickle

#生成训练数据
data = np.loadtxt('data.txt')
labels=np.loadtxt('labels.txt')
tdata = np.loadtxt('tdata.txt')
tlabels=np.loadtxt('tlabels.txt')
print(np.size(data,0),np.size(data,1))
#print(np.size(labels,0),np.size(labels,1))
print(np.size(tdata,0),np.size(tdata,1))
#print(np.size(tlabels,0),np.size(tlabels,1))
#print(labels)
#clf = svm.SVC(C=0.525, kernel='linear', gamma=8, decision_function_shape='ovo')
clf = svm.SVC(C=1, kernel='linear', gamma=20, decision_function_shape='ovo')
clf.fit(data, labels)
s=pickle.dumps(clf)
f=open('svm.model', "wb+")
f.write(s)
f.close()
f2=open('svm.model','rb')
s2=f2.read()
svm_model=pickle.loads(s2)
print(svm_model.predict(tdata))

#定义一个深度神经网络，324个输入节点，带有1个隐藏层，隐藏层由50个神经元组成，输出层由一个神经元组成
#multilayer_net = nl.net.newff([[0,255]]*324,[50,1])

#设置训练算法为梯度下降法
#multilayer_net.trainf = nl.

#训练网络
#multilayer_net.train.train_gdx(data,labels,show=100)

#用训练数据运行该网络，预测结果
#predicted_output=multilayer_net.sim(tdata)
#error=abs(predicted_output-tlabels)
#print(clf.score(data, labels))
#print(clf.score(tdata, tlabels))
#print(clf.predict(data))
