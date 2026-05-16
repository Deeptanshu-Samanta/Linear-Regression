import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradient_descent(x,y,w,b,alpha):
    y_pred=w*x+b
    dw=-(x*(y-y_pred))
    db=-(y-y_pred)
    w=w-alpha*dw
    b=b-alpha*db
    return w,b

data=pd.read_csv("https://raw.githubusercontent.com/Deeptanshu-Samanta/Linear-Regression/refs/heads/main/Experience-Salary.csv")
x=data["exp(in months)"]
y=data["salary(in thousands)"]
x=np.array(x)
y=np.array(y)
m=len(x)

plt.plot(x,y,"ro")  

w=0.0
b=0.0
alpha=0.001
y_pred=w*x+b
epochs=10000
for epoch in range(epochs):
    for i in range(0,m):
        w,b=gradient_descent(x[i],y[i],w,b,alpha)
        if(epoch%100==0):
            print(w,b)

y_pred=w*x+b

plt.plot(x,y_pred,"b-")
print("w=",w)
print("b=",b)

acc=np.zeros(m)
for i in range(0,m):
    a=y_pred[i]/y[i]
    if(a>1):
        a=2-a
    acc[i]=a*100

print("accuracy=",sum(acc)/m)

plt.show()
