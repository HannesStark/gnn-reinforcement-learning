import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme()

input_paths = [
    # 'Mlp_AntBulletEnv-v0_10-01_09-58-23',
    'GNN_PPO_inp_64_pro_64324_pol_64_val_64_64_N2048_B256_lr2e-04_mode_action_per_controller_Epochs_10_Nenvs_4_GRU_AntBulletEnv-v0_10-03_11-25-40',
    # 'GNN_PPO_inp_12_pro_64_64_64_64_pol_64_val_32_N2048_B64_lr3e-04_mode_flattened_Epochs_10_Nenvs_1_AntBulletEnv-v0_10-03_12-41-43',
    'nervenet_v1',
    'antgnn',

    #
    'MLP_TUNED_PPO_pi64_64_vf64_64_N2048_B64_lr2e-04_GNNValue_0_EmbOpt_shared_AntBulletEnv-v0_02-03_10-45-07',
    # 'NerveNetV2_GAT-GNN_PPO_inp_16_pro_161_161_161_161_pol_64_val_64_64_N2048_B64_lr2e-04_GNNValue_0_EmbOpt_sharedEpochs_10_Nenvs_1_GAT_AntBulletEnv-v0_07-03_23-58-23',

]

lists = []

for path in input_paths:
    with open(os.path.join('runs', path, 'rollout_mean.txt')) as file:
        lines = file.readlines()
        rewards = []
        for line in lines:
            rewards.append(float(line.strip('[').strip(']\n')))

        lists.append(rewards)

min_len = np.inf
for reward_list in lists:
    if len(reward_list) < min_len:
        min_len = len(reward_list)

arrays = []
for reward_list in lists:
    print(len(reward_list))
    arrays.append(np.array(reward_list)[:min_len])

# take the rolling mean by sliding an equally weighted window over the sequence
window_size = 100
for i, array in enumerate(arrays):
    arrays[i] = np.convolve(array, np.ones(window_size) / window_size, mode='valid')

plt.rcParams['figure.dpi'] = 300
array = np.array(arrays).T
df = pd.DataFrame(array, columns=range(len(arrays)))

sns.lineplot(data=df)
# plt.xlim([0, 2000])
plt.xlabel('number rollouts with 1024 steps per rollout')
plt.ylabel('mean reward in rollout')
# plt.title('Ant environment')
plt.legend(labels=['NerveNet-v2',
                   'NerveNet-v1',
                   'NerveNet-v0',
                   'MLP',
                   # 'MLP tuned'
                   ])
plt.savefig('ant.png')
plt.show()
plt.clf()
