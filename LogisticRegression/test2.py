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
    
    test_X = np.insert(test_X,121,0,axis=1)
    
    train_y2 = (train_y + 1)/2
    test_y2 = (test_y + 1)/2

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
        
    def getLoss(self,X,Y,f,method='l1'):
        #X: samples * features
        #W: features * 1
        #Y: samples * 1
        if method == 'l1':
            temp1 = np.dot(X,self.W) # samples * 1
            temp2 = np.multiply(Y,temp1) # samples * 1
            pureLoss = np.mean(np.log(1 + np.exp(-temp2))) # single value
            totalLoss = pureLoss + self.lambd * np.sum(np.square(self.W))/2
        if method == 'l2':
            temp1 = Y * np.log(self.sigmoid(X))
            temp2 = (1-Y)*(np.log(1-self.sigmoid(X)))
            pureLoss = np.mean(temp1+temp2)
            totalLoss = pureLoss
        f.write('pure Loss: '+str(pureLoss)+' total Loss: '+str(totalLoss)+'\n')
        return pureLoss, totalLoss
    
    def randomChoice(self,X,y,size=32):
        num = np.random.choice(len(X)-size-1)
        sample_x = X[num:num+size]
        sample_y = y[num:num+size]
        return sample_x,sample_y
        
    
    def trainWithSGD(self,X_train,y_train,X_val,y_val,batch_size=32,learning_rate=0.01,filename='test1',maxLoop=5):
        #y \in {+1,-1}
        f = open(filename,'w')
        f.write('Start training...\n')
        start = time.time()
        train_losses = []
        val_losses = []
        X_train = np.insert(X_train,0,1,axis=1)
        X_val = np.insert(X_val,0,1,axis=1)
        for i in range(maxLoop):
            X,y = self.randomChoice(X_train,y_train,size=batch_size)
            temp1 = (1-learning_rate*self.lambd)*self.W #features * 1
            temp2 = np.multiply(y,np.dot(X,self.W)) #samples * 1
            temp3 = np.multiply(y,X).T #features*samples
            temp4 = np.dot(temp3,1/(1 + np.exp(-temp2))) #features * 1
            self.W = temp1 + (learning_rate/batch_size)*temp4
            train_loss = self.getLoss(X_train,y_train,f)
            val_loss = self.getLoss(X_val,y_val,f)
            score = self.acc(y_val,self.predict(X_val))
            f.write('\nepoch '+str(i+1)+'   Training loss:   '+str(train_loss)+'  Valing loss:   '+str(val_loss)+' acc: '+str(score)+'\n\n')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        print(time.time()-start)
        f.close()
        return train_losses,val_losses
            
    
    def sigmoid(self,x):
        return 1 / (1.0 + np.exp(-np.dot(x,self.W)))
    
    def trainWithSGD2(self,X_train,y_train,X_val,y_val,batch_size=32,learning_rate=0.01,filename='test1',maxLoop=500):
        #y \in {+1,0}
        f = open(filename,'w')
        f.write('Start training...\n')
        start = time.time()
        train_losses = []
        val_losses = []
        X_train = np.insert(X_train,0,1,axis=1)
        X_val = np.insert(X_val,0,1,axis=1)
        for i in range(maxLoop):
            X,y = self.randomChoice(X_train,y_train,size=batch_size)
            temp1 = self.sigmoid(X)-y
            #按列求均值
            temp2 = np.mean(temp1*X,axis=0) #1*f
            self.W = self.W - learning_rate * temp2.T
            train_loss = self.getLoss(X_train,y_train,f,'l2')
            val_loss = self.getLoss(X_val,y_val,f,'l2')
            
            f.write('\nepoch '+str(i+1)+'   Training loss:   '+str(train_loss)+'  Valing loss:   '+str(val_loss)+'\n\n')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        print(time.time()-start)
        f.close()
        return train_losses,val_losses
        
        
    
    def trainWithAdam():
        pass
        
    
    def predict(self,X_test):
        X_test = np.insert(X_test,0,1,axis=1)
        prob = self.sigmoid(X_test)
        pre = np.where(prob>0.5,1,0)
        return pre
    
    def acc(self,y_tru,y_pre):
        count = 0
        for i in range(len(y_tru)):
            if y_tru[i] == y_pre[i]:
                count+=1
        return float(count)/len(y_tru)
        