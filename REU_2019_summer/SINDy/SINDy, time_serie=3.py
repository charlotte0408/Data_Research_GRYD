#!/usr/bin/env python
# coding: utf-8

# In[7]:


from pySINDy.sindy import SINDy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[9]:


df = pd.read_csv('Intersection_Y1R1R2_Y1.csv')
dfrY1 = df.loc[:,'Section_a':'Section_t']
Y1 = dfrY1.convert_objects(convert_numeric=True).values
Y1.shape


# In[12]:


df = pd.read_csv('Intersection_Y1R1R2_R1.csv')
dfrY2 = df.loc[:,'Section_a':'Section_t']
R1= dfrY2.convert_objects(convert_numeric=True).values
R1.shape


# In[13]:


df = pd.read_csv('Intersection_Y1R1R2_R2.csv')
dfrY3 = df.loc[:,'Section_a':'Section_t']
R2 = dfrY3.convert_objects(convert_numeric=True).values
R2.shape


# In[15]:


List_index=[]
for i in range(2581):
    for j in range(8):
        If_true=True
        if np.isnan(Y1[i][j]):
            for k in range(len(List_index)):
                if i==List_index[k]:
                    If_true=False
            if If_true:
                List_index.append(i)        


for i in range(2581):
    for j in range(8):
        If_true=True
        if np.isnan(R1[i][j]):
            for k in range(len(List_index)):
                if i==List_index[k]:
                    If_true=False
            if If_true:
                List_index.append(i)        


for i in range(2581):
    for j in range(8):
        If_true=True
        if np.isnan(R2[i][j]):
            for k in range(len(List_index)):
                if i==List_index[k]:
                    If_true=False
            if If_true:
                List_index.append(i)
List_index.sort()
List_index      


# In[34]:


test_mat=np.zeros((8,3))
for j in range(8):
        test_mat[j][0]=Y1[0][j]
        test_mat[j][1]=R1[0][j]
        test_mat[j][2]=R2[0][j]
model = SINDy(name='my_sindy_model')
model.fit(test_mat, 1, poly_degree=2, cut_off=0.001)
model.coefficients


# In[35]:


np.shape(model.descriptions)


# In[45]:


coeff=np.zeros((45,8))
for i in range(2581):
    copy_mat=np.zeros((8,3))
    for j in range(8):
        copy_mat[j][0]=Y1[i][j]
        copy_mat[j][1]=R1[i][j]
        copy_mat[j][2]=R2[i][j]
    model = SINDy(name='my_sindy_model')
    model.fit(copy_mat, 1, poly_degree=2, cut_off=0.01)
    coeff=np.add(coeff,model.coefficients)
for m in range (45):
    for n in range (8):
        coeff[m][n]=coeff[m][n]/2581
        if coeff[m][n]<0.01 and coeff[m][n]>-0.01:
            coeff[m][n]=0
coeff


# In[60]:


coefficient=pd.DataFrame(data=coeff)
coefficient.columns=['u0','u1','u2','u3','u4','u5','u6','u7']
coefficient.index=['1','u0', 'u1', 'u5', 'u6', 'u2', 'u3', 'u7', 'u4', 'u5^{2}', 'u0u6','u1u6','u2u6', 'u3u6','u6^{2}','u5u6','u4u5','u0u7','u1u7','u2u7','u3u7','u4u7','u5u7','u4u6','u3u5','u0u5','u1u5','u0^{2}','u0u1', 'u1^{2}', 'u0u2','u1u2', 'u2^{2}','u0u3','u2u5','u1u3','u3^{2}','u0u4','u1u4','u2u4','u3u4','u4^{2}','u6u7','u2u3', 'u7^{2}']
coefficient.round(3)


# In[59]:


model.descriptions


# In[68]:


from scipy.integrate import odeint

def model(a,t):
    dadt = -0.098-0.013 * a-0.023*a*a
    return dadt
t = np.linspace(0,6)

a_4 = odeint(model,4,t)
a_3 = odeint(model,3,t)
a_2 = odeint(model,2,t)
a_1 = odeint(model,1,t)
plt.plot(t,a_4)
plt.plot(t,a_3)
plt.plot(t,a_2)
plt.plot(t,a_1)
plt.xlabel('time')
plt.ylabel('a(t)')
plt.show()


# In[70]:



def model(b,t):
    dbdt = -0.066-0.013 * b-0.030*b*b
    return dbdt

t = np.linspace(0,6)

b_4 = odeint(model,4,t)
b_3 = odeint(model,3,t)
b_2 = odeint(model,2,t)
b_1 = odeint(model,1,t)
plt.plot(t,b_4)
plt.plot(t,b_3)
plt.plot(t,b_2)
plt.plot(t,b_1)
plt.xlabel('time')
plt.ylabel('b(t)')
plt.show()


# In[71]:


def model(c,t):
    dcdt = -0.028-0.014 * c-0.032*c*c
    return dcdt

t = np.linspace(0,6)

c_4 = odeint(model,4,t)
c_3 = odeint(model,3,t)
c_2 = odeint(model,2,t)
c_1 = odeint(model,1,t)
plt.plot(t,c_4)
plt.plot(t,c_3)
plt.plot(t,c_2)
plt.plot(t,c_1)
plt.xlabel('time')
plt.ylabel('c(t)')
plt.show()


# In[72]:


def model(d,t):
    dddt = -0.072-0.010 * d-0.023*d*d
    return dddt

t = np.linspace(0,6)

d_4 = odeint(model,4,t)
d_3 = odeint(model,3,t)
d_2 = odeint(model,2,t)
d_1 = odeint(model,1,t)
plt.plot(t,d_4)
plt.plot(t,d_3)
plt.plot(t,d_2)
plt.plot(t,d_1)
plt.xlabel('time')
plt.ylabel('d(t)')
plt.show()


# In[74]:


def model(f,t):
    dfdt = -0.055-0.015*f*f
    return dfdt

t = np.linspace(0,6)

f_4 = odeint(model,4,t)
f_3 = odeint(model,3,t)
f_2 = odeint(model,2,t)
f_1 = odeint(model,1,t)
plt.plot(t,f_4)
plt.plot(t,f_3)
plt.plot(t,f_2)
plt.plot(t,f_1)
plt.xlabel('time')
plt.ylabel('f(t)')
plt.show()


# In[76]:


def model(g,t):
    dgdt = -0.039-0.021*g-0.030*g*g
    return dgdt

t = np.linspace(0,6)

g_4 = odeint(model,4,t)
g_3 = odeint(model,3,t)
g_2 = odeint(model,2,t)
g_1 = odeint(model,1,t)
plt.plot(t,g_4)
plt.plot(t,g_3)
plt.plot(t,g_2)
plt.plot(t,g_1)
plt.xlabel('time')
plt.ylabel('g(t)')
plt.show()


# In[77]:


def model(h,t):
    dhdt = -0.031-0.013*h*h
    return dhdt

t = np.linspace(0,6)

h_4 = odeint(model,4,t)
h_3 = odeint(model,3,t)
h_2 = odeint(model,2,t)
h_1 = odeint(model,1,t)
plt.plot(t,h_4)
plt.plot(t,h_3)
plt.plot(t,h_2)
plt.plot(t,h_1)
plt.xlabel('time')
plt.ylabel('h(t)')
plt.show()


# In[78]:


def model(t_,t):
    dt_dt = -0.018*t_*t_
    return dt_dt

t = np.linspace(0,6)

t_4 = odeint(model,4,t)
t_3 = odeint(model,3,t)
t_2 = odeint(model,2,t)
t_1 = odeint(model,1,t)
plt.plot(t,t_4)
plt.plot(t,t_3)
plt.plot(t,t_2)
plt.plot(t,t_1)
plt.xlabel('time')
plt.ylabel('t_(t)')
plt.show()


# In[ ]:




