import numpy as np

# No. of samples to generate is set to be 5.
p = np.linspace(0, 1, num=5)

def squaring_histogram(p):
    # Bucket: Type used is float.
    B = np.array(p, dtype='f')
    N = B.size
  
    mean = np.mean(B)
    
    Giver = np.arange(N)
    Probs = np.empty(N)
    Probs.fill(mean)
    
    while np.count_nonzero(B < mean):

        i = np.where(B < mean)[0][0]
        j = np.where(B > mean)[0][0]

        Giver[i] = j
        Probs[i] = B[i]
        
        B[j] = B[j] - (mean - B[i])
        B[i] = mean
        
    return Giver, Probs


import pandas as pd
# Getting data and squaring.

data = pd.read_csv('us_births_69_88.csv')
births = np.array(data['births'])

# Squaring time can be calculated.
# %timeit squaring_histogram(births)

# ---------------------------------------------------------------------------------
# With vector it works two times slower than with set.
# 
# indexes = np.arange(n)
# def find_repetition_using_vec(a): # 42 sec on csv data- slower (time complexity)
#     _, uniq_ind = np.unique(a, return_index=True)
# #     mask = np.ones_like(indexes, dtype=bool)
#     mask[uniq_ind] = False
#     return(indexes[mask][0])
# ---------------------------------------------------------------------------------

# Trying to find repetitions with set as comparing with vector shows set is faster.
def find_repetitions_set(a):
    s = set()
    n = 0
    for v in a:
        n += 1
        
        if v in s:
            return n
        else:
#           Updates the set with v.
            s |= {v} 
    return n

n = births.size
mean = np.mean(births)
# K is the number of samples to be generated.
K = 77

sample_days = np.random.randint(0, n-1, K)
sample_births = np.random.random_sample(K) * mean

Donor, Probs = squaring_histogram(births)

# Brutal for explained
days = []
for i in range(K):
    day = sample_days[i]
    day_prob = sample_births[i]
    
    if day_prob < Probs[day]:
        days.append(day)
    else:
        days.append(Donor[day])

# Vectorizing.
use_original_value = sample_births < Probs[sample_days]
# ~ is bitwise NOT in python.
use_donor_value = ~use_original_value
sample_days[use_donor_value] = Donor[sample_days[use_donor_value]]

# Finds the repetitions.
find_repetitions_set(sample_days)


import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import time
# %matplotlib inline

# Repetitions
samples = 10**5
K = 77

Donor, Probs = squaring_histogram(births)

# Start of alg
start = time.time()

sample_days = np.random.randint(0, n-1, (samples, K))
sample_births = np.random.random_sample((samples,K)) * mean

use_original_value = sample_births < Probs[sample_days]
use_donor_value = ~use_original_value

sample_days[use_donor_value] = Donor[sample_days[use_donor_value]]

result = np.apply_along_axis(find_repetitions_set, 1, sample_days)

# End of alg
end = time.time()

plt.hist(result, np.arange(K))
plt.show(block=True);
# To show how fast the algorithm works. Note it is about time before plotting.
print("#Samples = {} - runtime: {}s".format(samples, end-start))