import pandas as pd
import math
import matplotlib.pyplot as plt

# Importing the dataset
ds = pd.read_csv(r'6 - Reinforcement Learning\datasources\ads_ctr_optimisation_data.csv')

# Implement the UCB algorithm
#N = 10000
#N = 1000
N = 500
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

for i in range(0, N):
    ad = 0
    max_upper_bound = 0
    for j in range(0, d):
        if(numbers_of_selections[j] > 0):
            average_reward = sums_of_rewards[j] / numbers_of_selections[j]
            delta_i = math.sqrt(3/2 * math.log(i + 1) / numbers_of_selections[j])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if(upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = j
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = ds.values[i,ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Show results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of items each ad was selected')
plt.show()