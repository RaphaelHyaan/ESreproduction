import numpy as np
a = np.linspace(1,10000,5000)
b = np.linspace(0,9999,5000)
c  = np.zeros((2,5000))
c[0] = a
c[1] = b
d = np.zeros((25,2,200))
for i in range(2):
    d[:,i,:] = np.reshape(c[i],(25,200))



print()