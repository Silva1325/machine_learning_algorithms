import pandas as pd
import math
import matplotlib.pyplot as plt
import random 

# Importing the dataset
ds = pd.read_csv(r'6 - Reinforcement Learning\datasources\ads_ctr_optimisation_data.csv')

# Implementing Thompson Sampling
#N = 10000
#N = 5000
#N = 500
N = 400
d = 10
ads_selected = []
numbers_of_rewards_0 = [0] * d
numbers_of_rewards_1 = [0] * d
total_reward = 0

for i in range(0, N):
    ad = 0
    max_random = 0
    for j in range(0, d):

        alpha = numbers_of_rewards_1[j] + 1
        beta = numbers_of_rewards_0[j] + 1
        random_beta = random.betavariate(alpha, beta)

        if random_beta > max_random:
            max_random = random_beta
            ad = j  # FIXED

    ads_selected.append(ad)
    reward = ds.values[i, ad]

    # FIXED
    if reward == 0:
        numbers_of_rewards_0[ad] += 1
    else:
        numbers_of_rewards_1[ad] += 1

    total_reward += reward

# Show results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
