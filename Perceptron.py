import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import os, subprocess

class Perceptron:
	def __init__(self, N):
		#generate random data
		xA, xB, yA, yB = [rnd.uniform(-1, 1) for i in range(4)] #gera 2 pontos que formam uma reta que vai dividir os dados
		self.V = np.array([xB*yA - xA*yB, yB-yA, xA-xB])
		self.X = self.point_gen(N)

	def classify(W, data):
		s = int(np.sign(W.T.dot(data)))
		return s
		
	def point_gen(self, N):
		X = []
		for i in range(N):
			x1, x2 = [rnd.uniform(-1, 1) for i in range(2)] #gera pontos aleatórios
			x = np.array([1, x1, x2])
			s = int(np.sign(self.V.T.dot(x)))
			X.append((x, s))
		return X

	def plot(self, mispts=None, vec=None, save=False):
		fig = plt.figure(figsize=(5,5))
		plt.xlim(-1,1)
		plt.ylim(-1,1)
		V = self.V
		a, b = -V[1]/V[2], -V[0]/V[2]
		l = np.linspace(-1,1)
		plt.plot(l, a*l+b, 'k-')
		cols = {1: 'r', -1: 'b'}
		for x,s in self.X:
			plt.plot(x[1], x[2], cols[s]+'o')
		if mispts:
			for x,s in mispts:
				plt.plot(x[1], x[2], cols[s]+'.')
		if vec != None:
			aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
			plt.plot(l, aa*l+bb, 'g-', lw=2)
		if save:
			if not mispts:
				plt.title('N = %s' % (str(len(self.X))))
			else:
				plt.title('N = %s with %s test points' % (str(len(self.X)), str(len(mispts))))
				plt.savefig('p_N%s_it%s' % (str(len(self.X), str(it))), dpi=200, bbox_inches='tight') 

	def classification_error(self, W, data=None):
		if not data:
			data = self.X
		numPts = len(data)
		numErr = 0
		for x,s in data:
			if self.classify(W, x) != s:
				numErr += 1
		error = numErr / float(numPts)
		return error

	def trainingVanilla(self, it, save=False):
		#inicializa pesos aleatórios
		w = np.zeros(3)
		for i in xrange(it):
			msPt = []
			for Xi in self.X:
				vec = Xi[0]
				classification = Xi[1]
				out = int(np.sign(w.T.dot(vec)))
				if out != classification:
					w += classification*vec
					msPt.append((vec, classification))
			if save:
				self.plot(vec=w)
				plt.title('N = %s, Iteration %s' % (str(len(self.X)), str(i+1)))
				plt.savefig('p_N%s_it%s' % (str(len(self.X)), str(i+1)), dpi=200, bbox_inches='tight') 
		self.w = w
	def trainingAdaline(self, it, eta):
		w = np.array([rnd.uniform(-1, 1) for i in range(3)])
		for i in range(it):
			msPt = []
			for Xi in self.X:
				vec = Xi[0]
				classification = Xi[1]
				out = int(np.sign(w.T.dot(vec)))
				error = out - classification
				w += error*eta*vec



MCP = Perceptron(80)
MCP.trainingVanilla(20, save=True)
basedir = 'C:\Users\danilo.souza\Documents\UFPE\Neural Network Playground'
os.chdir(basedir)
pngs = [pl for pl in os.listdir(basedir) if pl.endswith('png')]
sortpngs = sorted(pngs, key=lambda a:int(a.split('it')[1][:-4]))
basepng = pngs[0][:-8]
[sortpngs.append(sortpngs[-1]) for i in range(4)]
comm = ("convert -delay 50 %s %s.gif" % (' '.join(sortpngs), basepng)).split()
proc = subprocess.Popen(comm, stdin = subprocess.PIPE, stdout = subprocess.PIPE)
(out, err) = proc.communicate()