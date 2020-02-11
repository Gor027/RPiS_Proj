# Plotting function f(k)
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

dist = lambda k: 1/(abs(k) * (abs(k)+1) * (abs(k)+2)) if k else 0.5
# -10 is included but 11 is excluded
a = np.arange(-10, 11)
b = np.vectorize(dist)(a)

plt.plot(a, b, c='b')
plt.scatter(a, b, s=30)
# plt.show()


# Writting function sampling from Y's distribution

import random
import math

def sampleValue():
	# P(k = 0) = 0.5
	# As the graph shows the symmetry for prob of k != 0
	# Then P(k > 0) = P(k < 0) = (1 - 0.5)/0.5
	# So lets simulate the prob using switcher and random function
	sim = {
		1: -1,
		2: 0,
		3: 0,
		4: 1
	}

	place = sim[random.randint(1, 4)]
	probSim = np.random.uniform() / 4
	cum_dist = 0
	k = 1

	while cum_dist <= probSim:
		# Invariant:
		cum_dist = cum_dist + 1/(k*(k+1)*(k+2))
		# k++
		k += 1

	return place * k

# N is 10.000, generating N values from described distribution
N = 10000
# Values array according to given distribution
# Calls sampleValue 10000 times
values = np.array([sampleValue() for i in range(N)])

# print(values)

# Computing Y_i mean as shown in formula
a = np.arange(N)
mean = np.cumsum(values) / (a+1)
plt.plot(a, mean)
# plt.show()
# From the plot it is obvious that the series converges to 0.

# Computting median
# a[:(i)] is slice notation which means items from start to i-1
# That is why we add 1 to get from start to i
median = [np.median(values[:(i+1)]) for i in a]
plt.plot(a, median)
# It is obvious that the Y-axis should range really close to 0.
# So, we can limit the Y-axis close to it.
plt.ylim([-0.000005,0.000005])
plt.show()