'''
Created on 7 Apr 2014

@author Carlo Meijer
'''

import numpy as np

TREESIZE = 256
NTREES   = 100

# iNode
class iNode:
	@staticmethod
	def _splitData(data, splitAttr, splitValue):
		left  = []
		right = []
		for i in range(data.shape[0]):
			if data[i,splitAttr] < splitValue:
				left.append([data[i,j] for j in range(data.shape[1])])
			else:
				right.append([data[i,j] for j in range(data.shape[1])])

		return (np.matrix(left), np.matrix(right))

	@staticmethod
	def _generateSplitValue(randomState,data,splitAttr):
		_min = float("infinity")
		_max = float("-infinity")
		for i in range(data.shape[0]):
			if data[i, splitAttr] < _min:
				_min = data[i,splitAttr]
			elif data[i, splitAttr] > _max:
				_max = data[i,splitAttr]
		return randomState.rand() * (_max - _min) + _min

	@staticmethod
	def c(n):
		if (n-1) <= 0:
			return 0.0
		else:
			return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (float(n  - 1)) / float(n))

	def pathLength(self, point, e = 0):
		if isinstance(self, exNode):
			return e + iNode.c(self.size)

		if point[self.splitAttr] < self.splitValue:
			return self.left.pathLength(point, e+1)
		else:
			return self.right.pathLength(point, e+1)

	@staticmethod
	def createFromData(randomState, data, l, e=0):
		if e >= l or data.shape[0] <= 1:
			return exNode(data.shape[0])

		splitAttr = int(randomState.rand() * data.shape[1])
		splitValue = iNode._generateSplitValue(randomState,data, splitAttr)

		dataLeft, dataRight = iNode._splitData(data, splitAttr, splitValue)

		#print ' : {0} - {1}'.format(dataLeft.shape, dataRight.shape)
		#if dataLeft.shape[1] != 22:
			#print dataLeft
		nodeLeft  = iNode.createFromData(randomState, dataLeft, l, e+1)
		nodeRight = iNode.createFromData(randomState, dataRight, l, e+1)

		return inNode(nodeLeft, nodeRight, splitAttr, splitValue)

# internal node
class inNode(iNode):
	def __init__(self, left, right, splitAttr, splitValue):
		self.left = left
		self.right = right
		self.splitAttr = splitAttr
		self.splitValue = splitValue

# external node
class exNode(iNode):
	def __init__(self, size):
		self.size = size

class iForest:
	# create sample data for trees
	def createSample(self, psi):
		# flag already taken values
		f = {}
		out = []
		for i in range(psi):
			while True:
				samplePoint = int(self.randomState.rand() * self.data.shape[0])
				if not (samplePoint in f):
					break
				#else:
					#print '{0} is in f'.format(samplePoint)
			f[samplePoint] = True

			out.append([self.data[samplePoint, i] for i in range(self.data.shape[1])])
		return np.matrix(out)

	def anomalyScore(self, point):
		Havg = 0.0
		for i in range(self.t):
			Havg += self.trees[i].pathLength(point)

		Havg /= self.t
		return 1.0 / pow(2, Havg / iNode.c(self.psi))

	def __init__(self, data, t = NTREES, psi = TREESIZE):
		self.nDim = data.shape[1]
		self.t = t
		self.psi = psi
		self.data = data
		# height limit
		self.l = int(np.ceil(np.log(psi) / np.log(2)))
		self.trees = []
		self.randomState = np.random.RandomState(seed=None)

		for i in range(t):
			d = self.createSample(psi)
			self.trees.append(iNode.createFromData(self.randomState, d, self.l))

		#return self
