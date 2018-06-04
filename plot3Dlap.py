from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

data = np.loadtxt("laps/prueba.txt")


fig = plt.figure()

ax = fig.gca(projection='3d')

ax.plot(data[:,0],data[:,1],data[:,2],'b')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('V')
plt.show()

fig = plt.figure()

ax = fig.gca(projection='3d')

ax.plot(data[:,0],data[:,1],data[:,3],'r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('W')
plt.show()