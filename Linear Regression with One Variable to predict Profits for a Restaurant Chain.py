#!/usr/bin/env python
# coding: utf-8

# In[31]:


#we are using numpy, matplotlib and utils as a helper cell for this problem
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


#cell for loading the data
x_train, y_train = load_data()


# In[33]:


#cell for observing the data (where x_train is the population of city in 10000s and y_train is the profit made by restaurant in units of 10000)
print("Type of x_train:",type(x_train))
print("First ten elements of x_train are:\n", x_train[:10]) 
print("Type of y_train:",type(y_train))
print("First ten elements of y_train are:\n", y_train[:10]) 


# In[34]:


#cell for measuring the size of data
print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))


# In[35]:


#cell for observing the data graphically using matplotlib (on y axis :- profit and on x axis :- populationper city)
plt.scatter(x_train, y_train, marker='x', c='r') 
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()


# In[36]:


#cell for computing the cost function 
def compute_cost(x, y, w, b): 
   
   
    m = x.shape[0] 
    total_cost = 0
    sum = 0 
    for i in range (m):
        sum = sum + (w*x[i] + b - y[i])**2
        
    total_cost = sum / (2*m)
    
    return total_cost
    
   


# In[37]:


initial_w = 3
initial_b = 2

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')
from public_tests import *
compute_cost_test(compute_cost)


# In[38]:


#cell for computing the gradient descent 
def compute_gradient(x, y, w, b): 
   
    
    
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    sum_1 = 0
    sum_2 = 0
    for i in range (m):
        sum_1 = sum_1 + ((w*x[i] + b - y[i])*x[i])
        
    dj_dw = sum_1 / m
    
    for j in range (m):
        sum_2 = sum_2 + (w*x[i] + b - y[i])
        
    dj_db = sum_2 / m
  
        
    return dj_dw, dj_db
    


# In[39]:


initial_w = 1
initial_b = 1

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

compute_gradient_test(compute_gradient)


# In[40]:


test_w = 0.2
test_b = 0.2
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)

print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)


# In[41]:


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
   
    m = len(x)
    
    
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  
    b = b_in
    
    for i in range(num_iters):

       
        dj_dw, dj_db = gradient_function(x, y, w, b )  

       
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        
        if i<100000:      
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

       
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing


# In[42]:



initial_w = 0.
initial_b = 0.


iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                     compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)


# In[43]:


m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b


# In[44]:


plt.plot(x_train, predicted, c = "b")
plt.scatter(x_train, y_train, marker='x', c='r') 
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')


# In[45]:


predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))

predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))


# In[ ]:




