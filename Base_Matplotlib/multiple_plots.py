#Working with multiple plots 
#https://youtu.be/yZTBMMdPOww?t=1845

import numpy as np 
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t)* np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.subplot(211) #2 rows, 1 row, first plot
plt.plot(t1, f(t1), 'bo', t2, f(t2))

plt.subplot(212) # 2 rows, 1 column, second plot
plt.plot(t2, np.cos(2*np.pi*t2))
plt.show()