import matplotlib.pyplot as plt
import numpy as np
plt.ion() ## Note this correction
fig=plt.figure()
plt.axis([0,100,0,1])

i=0
x=list()
y=list()
z=list()

while i <100:
    temp_y=np.random.random() 
    temp_z=np.random.random() 
    x.append(i) 
    y.append(temp_y) 
    z.append(temp_z)
    plt.plot(x,y,'bo-',x,z,'k-')
    i+=1 
    plt.show()
    plt.pause(0.0001) #Note this correction