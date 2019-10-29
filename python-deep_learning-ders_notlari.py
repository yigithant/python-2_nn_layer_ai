# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
# a�a��daki ad�mlar s�ras�yla takip edilir


# Size of layers and initializing parameters weights and bias
# Forward propagation
# Loss function and Cost function
# Backward propagation
# Update Parameters
# Prediction with learnt parameters weight and bias
# Create Model

xx=np.load("/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy")
yy=np.load("/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy")

# xx ve yy adl� de�i�kenlere datalar aktar�ld�. 


import matplotlib.pyplot as plt

plt.imshow(x[100])
xx.shape

# %% [code]
x=np.concatenate((xx[204:409],xx[822:1027]),axis=0)
z=np.zeros(205)
o=np.ones(205)
y=np.concatenate((z, o), axis=0).reshape(x.shape[0],1)

# %% [code]
print("x shape : ",x.shape,"\ny shape : ",y.shape)

# %% [code]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3, random_state=42)

# %% [code]
x_train_t=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test_t=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

print("x_test_t shape : ", x_test_t.shape,"\nx_train_t shape : ",x_train_t.shape)

# %% [code]
# transpoze alma

x_train=x_train_t.T
x_test=x_test_t.T
y_train=y_train.T
y_test=y_test.T

# %% [code]
print("x_train shape : ", x_train.shape,"\nx_test shape : ",x_test.shape,"\ny_train shape : ", y_train.shape,"\ny_test shape : ", y_test.shape)

# %% [markdown]
# **Size of layers and initializing parameters weights and bias **

# %% [code]
# Size of layers and initializing parameters weights and bias X
# Forward propagation
# Loss function and Cost function
# Backward propagation
# Update Parameters
# Prediction with learnt parameters weight and bias
# Create Model

def initialize_parameters_weights_and_bias(x_train, y_train):
    param={"weight1":np.random.randn(3,x_train.shape[0])*0.1,
          "bias1":np.zeros((3,1)),
          "weight2":np.random.randn(y_train.shape[0],3)*0.1,
          "bias2":np.zeros((y_train.shape[0],1))}
    return param

# weight1: random olarka 3,4096'l�k bir numpy array olu�turuldu. 
# alttaki form�le uygun matris �arp�m�n� sa�l�cak bir matris elde edildi. random say�larla.
# randn fonkisyonu varyans� 1, ortalamas� 0 olan normal gauss da��l�m�nda rastgele say�lar �retiyor
# random say�lar� s�f�ra yak�n tutabilmek i�in 0.1 ile �arp�ld�.

# bias1 3 sat�r 1 s�tun array ile 0 matris olu�turuldu. 
# bias.shape=3,1 olan bir matris olu�turuldu. 

# weight2 1 sat�r 3 s�tunlu random bir matris olu�turuldu.

# bias2 1'e 1'lik bir array olu�turuldu. 0 matrisi. 

# z1=w1*x+b1
# a1=tanh(z1)
# z2=w2*a1+b2
# a2=sigma(z2)

# %% [markdown]
# ** Forward propagation**

# %% [code]
def sigmoid(X):
   return 1/(1+np.exp(-X))

# %% [code]
# Size of layers and initializing parameters weights and bias X
# Forward propagation X
# Loss function and Cost function
# Backward propagation
# Update Parameters
# Prediction with learnt parameters weight and bias
# Create Model


def forward_propagation(x_train,param):
    z1=np.dot(param['weight1'],x_train)+param['bias1']
    a1=np.tanh(z1)
    z2=np.dot(param['weight2'],a1)+param['bias2']
    a2=sigmoid(z2)
    cache={"z1":z1,"a1":a1,"z2":z2,"a2":a2}
    return a2,cache

# z1=w1*x+b1
# a1=tanh(z1)
# z2=w2*a1+b2
# a2=sigma(z2)

# yukardaki form�l forward_propagation ad�ndaki fonksiyon i�inde yap�ld�. 
# matrisler birbirleri dot �arp�ma u�rat�ld�. en son olarak cache ad�ndaki de�i�kenle fonksiyonda d�nd�rd�ld�.


# %% [markdown]
# **Loss function and Cost function **

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [code]
# Size of layers and initializing parameters weights and bias X
# Forward propagation X
# Loss function and Cost function X
# Backward propagation
# Update Parameters
# Prediction with learnt parameters weight and bias
# Create Model


def cost(a2, y, param):
    logprobs = np.multiply(np.log(a2),y)
    cost = -np.sum(logprobs)/y.shape[1]
    return cost

# yukardaki form�l fonksiyon i�erisinde tan�mland�. cost fonksiyonu  


# %% [markdown]
# **Backward propagation **

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [code]
# Size of layers and initializing parameters weights and bias X
# Forward propagation X
# Loss function and Cost function X
# Backward propagation X
# Update Parameters
# Prediction with learnt parameters weight and bias
# Create Model

#elde edilen aktivasyon fonksiyonu geriye d�n�k olarak g�ncellenmesi gerekir. bunun i�in backward
#propagation fonksiyonu ile yap�l�r. rastgele atanan weight ve bias de�erleri bu �ekilde geriye d�n�k olarak
#hatalar �zerinden e�itilir ve optimum de�erler elde edilmeye �al���r. geriye d�n�k t�revi al�nan 
#weight ve bias de�erleri elde edilir. bunlar daha sonra update edilmek �zere kullan�lacak olan de�erler.

def backward_propagation(param, cache, x, y):

    dZ2 = cache["a2"]-y
    dW2 = np.dot(dZ2,cache["a1"].T)/x.shape[1]
    db2 = np.sum(dZ2,axis =1,keepdims=True)/x.shape[1]
    dZ1 = np.dot(param["weight2"].T,dZ2)*(1 - np.power(cache["a1"], 2))
    dW1 = np.dot(dZ1,x.T)/x.shape[1]
    db1 = np.sum(dZ1,axis =1,keepdims=True)/x.shape[1]
    grads = {"dweight1": dW1,
             "dbias1": db1,
             "dweight2": dW2,
             "dbias2": db2}
    return grads

# %% [markdown]
# **Update Parameters **

# %% [code]
# Size of layers and initializing parameters weights and bias X
# Forward propagation X
# Loss function and Cost function X
# Backward propagation X
# Update Parameters X
# Prediction with learnt parameters weight and bias
# Create Model

# backward propagation ile elde etti�imiz de�erler ile yeni weight ve bias de�erleri olu�turulup bu de�erler
# ile g�ncellenmi� olacak. 
# learning rate ile ��renmenin h�z� belirlenecek. batch s�ras�nda g�r�lt�y� ��renmeye engellemek ama�l�
# kullan�l�r.

def update_param(param, grads, learning_rate=0.01):
    param={"weight1": param['weight1']-learning_rate*grads['dweight1'],
          "bias1":param['bias1']-learning_rate*grads['dbias1'],
          "weight2":param['weight2']-learning_rate*grads['dweight2'],
          "bias2":param["bias2"]-learning_rate*grads['dbias2']}
    return param




# %% [markdown]
# **Prediction with learnt parameters weight and bias**

# %% [code]
# Size of layers and initializing parameters weights and bias X
# Forward propagation X
# Loss function and Cost function X
# Backward propagation X
# Update Parameters X
# Prediction with learnt parameters weight and bias X
# Create Model

def prediction(param,x_test):
    a2,cache=forward_propagation(x_train,param)
    y_predict=np.zeros((1,x_test.shape[0]))
    for i in range(a2.shape[1]):
        if a2[0,i]<=0.5:
            y_predict[0,i]=0
        else:
            y_predict[0,i]=1
    return y_predict


# a2 ve cache de�i�kenleri forward_propagation fonksiyonuna verildi.
# y_preditc ad�nda bir de�i�kene 1'e x_test.shape[0]'lik bir matris olu�turuldu.
# ard�ndan elde edilen a2 de�eri e�er 0.5 den k���k veya e�itse 0 olarak verldi.
# de�ilse a2 de�eri 1 olarak verildi.
# y_predict olarak fonksiyon d�nd�r�ld�.


# %% [markdown]
# **Create Model**

# %% [code]
# Size of layers and initializing parameters weights and bias X
# Forward propagation X
# Loss function and Cost function X
# Backward propagation X
# Update Parameters X
# Prediction with learnt parameters weight and bias X
# Create Model X

# model olu�turulmas� yukar�daki ad�mlar�n s�rayla uygulanmas�yla elde edilir.bir kere parametreler
# initialize edilir ard�ndan ne kadar iterasyon yap�lacaksa di�er ad�mlar s�rayla iterasyona g�re tekrarlan�r

def create_model(x_train,y_train,x_test,y_test,num_iteration):
    parameters=initialize_parameters_weights_and_bias(x_train,y_train)
    cost_list=[]
    index_list=[]
    for i in range(0,num_iteration):
        a2,cache=forward_propagation(x_train,parameters)
        cost_f=cost(a2,y_train,parameters)
        grads_g=backward_propagation(parameters,cache,x_train,y_train)
        parameters=update_param(parameters,grads_g)
        
        if i%100==0:
            print(i,". cost : ",cost_f)
            cost_list.append(cost_f)
            index_list.append(i)
        
    
    plt.plot(index_list,cost_list)
    y_prediction_test=prediction(parameters,x_test)
    y_prediction_train=prediction(parameters,x_train)
    
        

# %% [code]
parameters=create_model(x_train,y_train,x_test,y_test,2500)

# %% [code]
#kaynak : https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners#Artificial-Neural-Network-(ANN)