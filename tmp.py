import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme()

input_paths = [
    'Mlp_AntBulletEnv-v0_09-01_10-29-48',
    'Mlp_HopperBulletEnv-v0_09-01_10-30-04'
]

lists = []
min_len = np.inf
for path in input_paths:
    with open(os.path.join('runs', path, 'results.txt')) as file:
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

plt.rcParams['figure.figsize'] = [10, 8]
array = np.array(arrays).T
df = pd.DataFrame(array, columns=range(len(arrays)))

sns.lineplot(data=df)
plt.xlabel('number of steps')
plt.ylabel('mean reward in rollout')
plt.legend(labels=['reacher', 'hopper'])
# plt.savefig('figure.png')
plt.show()
plt.clf()
