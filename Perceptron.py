import numpy as np
import random as rnd

class Perceptron:
	def __init__(self, N):
		#generate random data
		xA, xB, yA, yB = [rnd.uniform(-1, 1) for i in range(4)] #gera 2 pontos que formam uma reta que vai dividir os dados
		self.V = np.array([xB*yA - xA*yB, yB-yA])
		self.X = self.point_gen(N)

	def point_get(self, N):
		X = []
		for i in range(N):
			x1, x2 = [random.uniform(-1, 1) for i in range(2)] #gera pontos aleatórios
			x = np.array([1, x1, x2])
			s = int(np.sign(self.V.T.dot(x)))
			X.append((x, s))
		return X

	def trainingVanilla(self, iter):
		step_function = lambda x: 0 if x < 0 else 1
		#inicializa pesos aleatórios
		w = rnd.rand(3)
		for i in xrange(iter):
			for Xi in X:
				vec = Xi[0]
				classification = Xi[1]
				out = step_function(np.dot(vec, w))
				if out != classification:
					w += classification*vec

	def trainingAdaline(self, iter, eta):
