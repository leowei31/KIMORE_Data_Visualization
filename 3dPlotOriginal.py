import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

# import muscle activity postion csv
muscle = np.genfromtxt("JointPositionExercise4.csv", delimiter = ',', skip_header = 1)

# loop through columns and for each column, subtract the lumbar original position from each to set is as an absoulte position
for k in range(4, 99, 4):
    muscle[:, k] = muscle[:, k] - np.median(muscle[:,0])
    muscle[:, k+1] = muscle[:, k+1] - np.median(muscle[:,1])
    muscle[:, k+2] = muscle[:, k+2] - np.median(muscle[:,2])

muscle[:, 0] = muscle[:, 0] - np.median(muscle[:, 0])
muscle[:, 1] = muscle[:, 1] - np.median(muscle[:, 1])
muscle[:, 2] = muscle[:, 2] - np.median(muscle[:, 2])

# transpose the data because the data was imported as columns and we need them as row vectors
# list1 = np.array([1,2,3,4,5,6,7,8,9])

# for num in range(2,8,2):
#     print(list1[num])

# dictionary = {'low': 1, 'medium':2, 'high':3}
# print(dictionary['medium'])

fig = plt.figure()
ax = fig.gca(projection = '3d')

ax.plot(np.transpose(muscle[:,0]),np.transpose(muscle[:,1]),np.transpose(muscle[:,2]), color = 'red')

for k in range(4, 99, 4):
    ax.plot(np.transpose(muscle[:,k]), np.transpose(muscle[:,k+1]), np.transpose(muscle[:,k+2]),color = 'red', label = 'movement')
# ax.plot(np.transpose(muscle[:,4]), np.transpose(muscle[:,5]), np.transpose(muscle[:,6]),color = 'red', label = 'movement')
plt.show()