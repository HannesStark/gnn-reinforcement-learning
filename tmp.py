import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme()

input_paths = [
    'Mlp_AntBulletEnv-v0_10-01_09-58-23'
]

lists = []
min_len = np.inf
for path in input_paths:
    with open(os.path.join('runs', path, 'rollout_mean.txt')) as file:
        lines = file.readlines()
        rewards = []
        for line in lines:
            rewards.append(float(line))
        if len(rewards) < min_len:
            min_len = len(rewards)
        lists.append(rewards)
arrays = []
for reward_list in lists:
    arrays.append(np.array(reward_list)[:min_len])

# take the mean
window_size=100
for i, array in enumerate(arrays):
    arrays[i] = np.convolve(array,np.ones(window_size)/window_size,mode='valid')

plt.rcParams['figure.figsize'] = [10, 8]
array = np.array(arrays).T
df = pd.DataFrame(array, columns=range(len(arrays)))

sns.lineplot(data=df)
plt.xlabel('1000 number of steps')
plt.ylabel('mean reward in rollout')
plt.legend(labels=['MLP', 'NerveNet-MLP'])
# plt.savefig('figure.png')
plt.show()
plt.clf()
