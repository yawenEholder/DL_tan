import numpy as np
import time
def main():
    from sklearn.datasets import load_svmlight_file
    train_data = load_svmlight_file("a9a_train.txt")
    test_data = load_svmlight_file("a9a_test.txt")
    
    train_X,train_y = train_data[0],train_data[1]
    test_X,test_y = test_data[0],test_data[1]
    train_X = train_X.toarray()
    test_X = test_X.toarray()
    train_y = train_y.reshape((len(train_y),1))
    test_y = test_y.reshape((len(test_y),1))

class LogisticRegression(object):
    def __init__(self):
        self.W = 0
        self.lambd = 0
    
    def initParameters(self,shape,method='random'):
        if method == 'default':
            #add one dimension of b
            self.W = np.zeros(shape+1)
        elif method == 'random':
            self.W = np.random.rand(shape+1)
        elif method == 'norm':
            self.W = np.random.randn(shape+1)
        self.W = self.W.reshape((len(self.W),1))
        
    def getLoss(self,X,Y):
        #X: features * samples
        #W: features * 1
        #Y: samples * 1
        temp1 = np.dot(self.W.T,X) # 1 * samples
        temp2 = np.multiply(Y,temp1.T) # samples * 1
        pureLoss = np.mean(np.log(1 + np.exp(-temp2))) # single value
        totalLoss = pureLoss + self.lambd * np.dot(self.W.T,self.W)/2
        print('pure Loss: '+str(pureLoss)+' total Loss: '+str(totalLoss))
        return  totalLoss
    
    def randomChoice(self,X,y,size=32):
        num = np.random.choice(len(X)-size-1)
        sample_x = X[num:num+size]
        sample_y = y[num:num+size]
        return sample_x,sample_y
        
    
    def trainWithSGD(self,X_train,y_train,X_val,y_val,batch_size=32,learning_rate=0.01,filename='test1',maxLoop=500):
        f = open(filename,'w')
        f.wtite('Start training...\n')
        start = time.time()
        train_losses = []
        val_losses = []
        for i in maxLoop:
            X,y = randomChoice(X_train,y_train,size=batch_size)
            temp1 = (1-learning_rate*self.lambd)*self.W #features * 1
            temp2 = np.multiply(y,np.dot(self.W.T,X).T) #samples * 1
            temp3 = np.multiply(y,X.T).T #features*samples
            temp4 = np.dot(temp3,1/(1 + np.exp(-temp2))) #features * 1
            self.W = temp1 + (learning_rate/batch_size)*temp4
            y_tr
            
        
        
        
        
    
    def trainWithAdam():
        pass
        
    
    def predict(self,X_test):
        #X_test: features * samples
        X_test = np.insert(X_test,0,1,axis=1)
        y_pre = np.dot(X_test,self.W)
        return y_pre