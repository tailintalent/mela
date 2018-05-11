import numpy as np
import matplotlib.pylab as plt


x = [1,2,3,4,5,6,7,8,9,10,11]
y = [0.00197234,0.0018456,0.001764,0.00168665,0.00161332,0.00154376,0.00147775,0.00141508,0.00135557,0.00129904,0.00124532]
 
plt.plot(x,y)
plt.title("K-Shot learning (K=10), tanh")
plt.xlabel("# Gradient Steps")
plt.ylabel("MSE")
plt.show()