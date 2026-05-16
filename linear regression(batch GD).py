import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradient_descent(x,y,w,b,alpha):
    m=len(x)
    y_pred=w*x+b
    dw=(-1/m)*sum(x*(y-y_pred))
    db=(-1/m)*sum(y-y_pred)
    w=w-alpha*dw
    b=b-alpha*db
    return w,b

data=pd.read_csv("C:\\Users\\deept\\OneDrive\\Desktop\\ml_models\\Experience-Salary.csv")
x=data["exp(in months)"]
y=data["salary(in thousands)"]
x=np.array(x)
y=np.array(y)
m=len(x)

plt.plot(x,y,"ro")

w=np.zeros(1)
b=0
alpha=0.001
y_pred=w*x+b
costi=(1/(2*m))*sum((y-y_pred)**2)

while True:
    w,b=gradient_descent(x,y,w,b,alpha)
    y_pred=w*x+b
    costf=(1/(2*m))*sum((y-y_pred)**2)
    if(costi<costf):
        break
    costi=costf

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