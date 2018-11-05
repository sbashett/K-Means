import numpy as np
import random
import matplotlib.pyplot as plt

data = np.loadtxt('./data.txt', delimiter = ',')

data = np.delete(data, 0,1)
data = np.delete(data,9,1 )

print(data.shape)

k = [1,2,3,4,5,6,7,8,9,10]
graph = np.zeros(len(k))

for it in k:
	centroids = np.zeros([it, data.shape[1]])
	for i in range(it):
		lower = random.randint(1, (int)(data.shape[0]/2))
		upper = random.randint((int)(data.shape[0]/2), data.shape[0]-1)
		centroids[i] = np.average(data[lower:upper], axis = 0)

	distances = np.zeros([data.shape[0], it])

	temp = np.zeros(centroids.shape)

	while np.sum(centroids - temp) != 0:

		temp[:,:] = centroids[:,:]

		for i in range(it):
			distances[:,i] = np.linalg.norm((data - centroids[i]), axis = 1)

		classify = np.argsort(distances, axis = 1)
		classify = np.delete(classify,np.arange(1,it),1)

		arrange = np.reshape(np.argsort(classify, axis = 0), data.shape[0])

		lower = 0
		potential = 0
		for i in range(data.shape[0]-1):
			if classify[arrange[i]] != classify[arrange[i+1]]:
				upper = i+1
				centroids[classify[arrange[i]]] = np.average(data[arrange[lower:upper]], axis = 0)
				potential += np.sum(np.square(np.linalg.norm(data[arrange[lower:upper]]-centroids[classify[arrange[i]]], axis = 1)))
				#print([lower,upper])
				lower = upper

		centroids[classify[arrange[i-1]]] = np.average(data[arrange[lower:data.shape[0]]], axis = 0)
		potential += np.sum(np.square(np.linalg.norm(data[arrange[lower:data.shape[0]]]-centroids[classify[arrange[i-1]]], axis = 1)))
	graph[k.index(it)] = potential

print(graph)

# plotting the graphs on same plane for different k values and train and test errors
plt.plot(k,graph)
plt.xlabel('k value')
plt.ylabel('potential')
plt.show()	