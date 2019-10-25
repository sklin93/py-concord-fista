from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np

#variable n should be number of curves to plot (I skipped this earlier thinking that it is obvious when looking at picture - sorry my bad mistake xD): n=len(array_of_curves_to_plot)
#version 1:

# color=cm.rainbow(np.linspace(0,1,n))
# for i,c in zip(range(n),color):
#    ax1.plot(x, y,c=c)

# #or version 2: - faster and better:

# color=iter(cm.rainbow(np.linspace(0,1,n)))
# c=next(color)
# plt.plot(x,y,c=c)

#or version 3:

n = 5
color=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(n):
   c=next(color)
   x = range(10)
   y = 
   plt.plot(x, y,c=c)