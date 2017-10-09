from __future__ import print_function, division
from future import standard_library
import numpy as np


""" Super Class """
class Optimizer(object):
	""" 
	This is a template for implementing the classes of optimizers
	"""
	def __init__(self, net, lr=1e-4):
		self.net = net  # the model
		self.lr = lr    # learning rate

	""" Make a step and update all parameters """
	def step(self):
		for layer in self.net.layers:
			for n, v in list(layer.params.items()):
				pass


""" Classes """
class SGD(Optimizer):
	""" Some comments """
	def __init__(self, net, lr=1e-4):
		self.net = net
		self.lr = lr

	def step(self):
		for layer in self.net.layers:
			for n, v in list(layer.params.items()):
				dv = layer.grads[n]
				layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
	def __init__(self, net, lr=1e-4, momentum=0.0):
		self.net = net
		self.lr = lr
		self.momentum = momentum
		self.velocity = {k: np.zeros_like(v) for layer in self.net.layers for k, v in layer.params.items()}

	def step(self):
		#############################################################################
		# TODO: Implement the SGD + Momentum                                        #
		#############################################################################
		for layer in self.net.layers:
			for n in layer.params.keys():
				dv = layer.grads[n]
				self.velocity[n] = self.momentum * self.velocity[n] - self.lr * dv
				layer.params[n] += self.velocity[n]
				# print(self.velocity[n])
				# print(layer.params[n])
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################


class RMSProp(Optimizer):
	def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
		self.net = net
		self.lr = lr
		self.decay = decay
		self.eps = eps
		self.cache = {k: np.zeros_like(v) for layer in self.net.layers for k, v in layer.params.items()}  # decaying average of past squared gradients

	def step(self):
		#############################################################################
		# TODO: Implement the RMSProp                                               #
		#############################################################################
		# pass
		for layer in self.net.layers:
			for n in layer.params.keys():
				dv = layer.grads[n]
				self.cache[n] = self.decay * self.cache[n] + (1-self.decay) * np.power(dv, 2)
				layer.params[n] -= self.lr * dv / np.sqrt(self.cache[n] + self.eps)
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################


class Adam(Optimizer):
	def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
		self.net = net
		self.lr = lr
		self.beta1, self.beta2 = beta1, beta2
		self.eps = eps
		self.mt = {k: np.zeros_like(v) for layer in self.net.layers for k, v in layer.params.items()}
		self.vt = {k: np.zeros_like(v) for layer in self.net.layers for k, v in layer.params.items()}
		self.t = t

	def step(self):
		#############################################################################
		# TODO: Implement the Adam                                                  #
		#############################################################################
		# pass
		self.t += 1
		for layer in self.net.layers:
			for n in layer.params.keys():
				dv = layer.grads[n]
				self.mt[n] = self.beta1 * self.mt[n] + (1-self.beta1) * dv
				self.vt[n] = self.beta2 * self.vt[n] + (1-self.beta2) * np.power(dv, 2)

				# Bias correct
				mt_bias = self.mt[n] / (1-np.power(self.beta1, self.t))
				vt_bias = self.vt[n] / (1-np.power(self.beta2, self.t))

				layer.params[n] -= self.lr * mt_bias / (np.sqrt(vt_bias) + self.eps)
				# print(self.mt[n])
		
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################