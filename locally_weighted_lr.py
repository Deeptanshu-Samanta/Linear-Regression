import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MASS/mcycle.csv")

def weight(x,x_in,tau):
    w=np.exp(-((x-x_in)**2)/(2*tau**2))
    return w

def gradient_descent(x,y,m,alpha,w,theta,b):
    y_pred=theta*x+b
    dw=(-1/m)*sum(w*x*(y-y_pred))
    db=(-1/m)*sum(w*(y-y_pred))
    theta=theta-alpha*dw
    b=b-alpha*db
    return theta,b


x=np.array(data["times"])
y=np.array(data["accel"])
plt.plot(x,y,"ro")

x_in=float(input("Enter the time to predict the acceleration(2.5 to 57.5): "))

tau=2.0
w=weight(x,x_in,tau)
theta=0
b=0
m=len(x)
alpha=0.001
y_pred=theta*x+b
costi=(1/(2*m))*sum(w*(y-y_pred)**2)

while True:
    theta,b=gradient_descent(x,y,m,alpha,w,theta,b)
    y_pred=theta*x+b
    costf=(1/(2*m))*sum(w*(y-y_pred)**2)
    if(costi<costf):
        break
    costi=costf

print("w=",theta)
print("b=",b)

y_p=theta*x_in+b
print("Predicted acceleration at time",x_in,"is",y_p)

plt.plot(x_in,y_p,"go")
plt.show()
