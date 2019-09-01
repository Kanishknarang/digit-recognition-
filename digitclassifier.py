#importing libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np 
import matplotlib.pyplot as plt

#loading digits data
digits = load_digits()
#print(digits.data[0])
#print(digits.images[0])

f, axrr = plt.subplots(2,3)
axrr[0,0].imshow(digits.images[0])
axrr[0,1].imshow(digits.images[1])
axrr[0,2].imshow(digits.images[2])
plt.show()

class LogisticsRegression:
    def set_values(self,initial_params,alpha=0.01,max_iter=5000,class_of_interest=0):
        self.params = initial_params
        self.alpha = alpha
        self.max_iter = max_iter
        self.class_of_interest = class_of_interest
    
    @staticmethod
    def _sigmoid(x):
        return 1.0/(1+ np.exp(-x))

    def predict(self,x_bar,params):
        return self._sigmoid(np.dot(params,x_bar))

    def compute_cost(self,input_var, output_var, params):

        cost =0
        for x,y in zip(input_var,output_var):
            x_bar = np.array(np.insert(x,0,1))
            y_hat = self.predict(x_bar,params)

            y_binary = 1.0 if y==self.class_of_interest else 0.0

            cost+= y_binary * np.log(y_hat) + (1-y_binary)*np.log(1-y_hat)
        return cost

        
    def train(self,input_var,label,print_iter = 500):
        iteration = 1
        
        while iteration<=self.max_iter:
            if iteration % print_iter==0:
                print(self.compute_cost(input_var,label,self.params))
            for i,xy in enumerate(zip(input_var,label)):
                x_bar = np.array(np.insert(xy[0],0,1))
                y_hat = self.predict(x_bar,self.params)

                y_binary = 1.0 if xy[1]==self.class_of_interest else 0.0
                gradient = (y_binary-y_hat)*x_bar
                self.params+= self.alpha* gradient
            
            iteration+=1
        
        return self.params

    def test(self,input_var,label):
        self.total_values = 0
        self.correct_values = 0
        for i,xy in enumerate(zip(input_var,label)):
            self.total_values+=1
            x_bar = np.array(np.insert(xy[0],0,1))
            y_hat = self.predict(x_bar,self.params)
            y_binary = 1.0 if xy[1]==self.class_of_interest else 0.0

            if y_hat>=0.5 and y_binary==1:
                self.correct_values+=1
            elif y_hat<0.5 and y_binary==0:
                self.correct_values+=1
            
        return self.correct_values/self.total_values


#splitting data into traing and testing data set
digits_train,digits_test,digits_label_train,digits_label_test = train_test_split(digits.data,digits.target,test_size = 0.20)
print(len(digits_label_test))
alpha = 0.01
max_iter = 10000
params_0 = np.zeros(len(digits.data[0])+1)
#print(params_0)

params=[]
for i in range(10):
    digits_regression_model_0 = LogisticsRegression()
    digits_regression_model_0.set_values(params_0,alpha,max_iter,i)

    params.append(digits_regression_model_0.train(digits_train/16.0,digits_label_train,500))

    print(digits_regression_model_0.test(digits_test/16.0,digits_label_test))



total_values = 0
correct_values = 0
for i,xy in enumerate(zip(digits_test/16.0,digits_label_test)):
    total_values+=1
    x_bar = np.array(np.insert(xy[0],0,1))
    max = -1.0
    d = -1
    for j in params:
        prediction = LogisticsRegression._sigmoid(np.dot(j,x_bar))
        if prediction > max:
            max = prediction
            d = params.index(j)
    
    
    if d==digits_label_test[i]:
        correct_values+=1

print(correct_values/total_values)