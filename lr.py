import numpy as np
import time

#implement a base class of Linear Regression Model
class LinearRegression(object):
    def __init__(self):
        self.W = 0
        self.lamda = 1.0
        
    
    def initialPar(self,shape,method='default'):
        '''
        shape: the shape of par W
        mathod: the initial method of par W
        '''
        if method == 'default':
            #add one dimension of b
            self.W = np.zeros(shape+1)
        elif method == 'random':
            self.W = np.random.rand(shape+1)
        elif method == 'norm':
            self.W = np.random.randn(shape+1)
        
    def getLoss(self,y_pre,y_ground,lamda=1.0,method='squared'):
        '''
        method: the method of get Loss
        '''
        self.lamda = lamda
        if method == 'absolute':
            loss = np.sum(np.fabs(y_pre-y_ground)) / y_pre.shape[0]
        elif method == 'squared':
            loss = np.sum(np.square(y_pre-y_ground)) /(2* y_pre.shape[0])
        sloss = loss + np.sum(np.square(self.W))*lamda/2
        print('Pure loss: '+str(loss)+'.....Total loss: '+str(sloss))
        return loss
        
    def train(self,X_train,y_train):
        pass
    
    def predict(self,X_test):
        X_test = np.insert(X_test,0,1,axis=1)
        y_pre = np.dot(X_test,self.W)
        return y_pre
    
# implement a subclass of Linear Regression Model: Closed-Form Solution
class ClosedFormLinearRegression(LinearRegression):
    def __init__(self):
        super(ClosedFormLinearRegression,self).__init__()
        
    def train(self,X_train,y_train):
        start = time.time()
        print('Training...')
        X_train = np.insert(X_train,0,1,axis=1)
        print(X_train.shape)
        w_temp = np.dot(np.transpose(X_train),X_train)+self.lamda * np.identity(X_train.shape[1])
        #w_temp = np.dot(np.transpose(X_train),X_train)
        try:
            w_temp = np.linalg.inv(w_temp)
        except np.linalg.LinAlgError:
            print('Singular matrix!!!!!Process interprutted!!!!!')
        else:
            w_temp = np.dot(w_temp,np.transpose(X_train))
            w_temp = np.dot(w_temp,y_train)
        self.W = w_temp
        print('Training...'+str(time.time()-start)+'s...Successful!!')
        
#implement a subclass of Linear Regression Model: Gradient Descent Solution
class GradientDescentLinearRegression(LinearRegression):
    def __init__(self):
        super(GradientDescentLinearRegression,self).__init__()
    
    def train(self,X_train,y_train,X_val,y_val,maxLoop=500,epsilon=0.01,learning_rate = 0.1):
        loss = 10e10
        start = time.time()
        print('Training...')
        X_train = np.insert(X_train,0,1,axis=1)
        for i in range(maxLoop):
            count = np.random.choice(X_train.shape[0])
            x_sample = X_train[count]
            y_sample = y_train[count]
            g_temp1 = self.lamda*self.W - np.dot(x_sample.T,y_sample)
            g_temp2 = np.dot(np.dot(x_sample.T,x_sample),self.W)
            gradient =  g_temp1 + g_temp2
            self.W = self.W - learning_rate * gradient
            #learning_rate = learning_rate / (i+1)
            y_train_pre = np.dot(X_train,self.W)
            y_val_pre = self.predict(X_val)
            y_train_loss = self.getLoss(y_train_pre,y_train)
            y_val_loss = self.getLoss(y_val_pre,y_val)
            print('epoch '+str(i+1)+'   Training loss:   '+str(y_train_loss)+'  Valing loss:   '+str(y_val_loss))
            #if abs(loss - y_train_loss) > epsilon:
            #    loss = y_train_loss
            #else:
            #    print(str(loss-y_train_loss))
            #    print("Convergencing...")
            #    break
        print('Training...'+str(time.time()-start)+'s...Successful!!')
