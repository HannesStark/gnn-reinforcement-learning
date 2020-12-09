"""
    
    @author:
        Tobias Schmidt, Hannes Stark, modified from the code of Tingwu Wang.
"""

import num2words

TASK_DICT = {
    'Centipede': [3, 5, 7] + [4, 6, 8, 10, 12, 14] + [20, 30, 40, 50],
    'CpCentipede': [3, 5, 7] + [4, 6, 8, 10, 12, 14],
    'Reacher': [0, 1, 2, 3, 4, 5, 6, 7],
    'Snake': range(3, 10) + [10, 20, 40],
}

# walker list
MULTI_TASK_DICT = {
    'MultiWalkers-v1':
        ['WalkersHopper-v1', 'WalkersHalfhumanoid-v1', 'WalkersHalfcheetah-v1',
         'WalkersFullcheetah-v1', 'WalkersOstrich-v1'],
    # just for implementation, only one agent will be run
    'MultiWalkers2Kangaroo-v1':
        ['WalkersHopper-v1', 'WalkersHalfhumanoid-v1', 'WalkersHalfcheetah-v1',
         'WalkersFullcheetah-v1', 'WalkersKangaroo-v1'],
}

# test the robustness of agents
NUM_ROBUSTNESS_AGENTS = 5
ROBUSTNESS_TASK_DICT = {}
for i_agent in range(NUM_ROBUSTNESS_AGENTS + 1):
    ROBUSTNESS_TASK_DICT.update({
        'MultiWalkers' + num2words.num2words(i_agent) + '-v1':
            ['WalkersHopper' + num2words.num2words(i_agent) + '-v1',
             'WalkersHalfhumanoid' + num2words.num2words(i_agent) + '-v1',
             'WalkersHalfcheetah' + num2words.num2words(i_agent) + '-v1',
             'WalkersFullcheetah' + num2words.num2words(i_agent) + '-v1',
             'WalkersOstrich' + num2words.num2words(i_agent) + '-v1'],
    })
MULTI_TASK_DICT.update(ROBUSTNESS_TASK_DICT)


def get_mujoco_model_settings():
    '''
        @brief:
            @traditional environments:
                1. Humanoid-v1
                2. HumanoidStandup-v1
                3. HalfCheetah-v1
                4. Hopper-v1
                5. Walker2d-v1
                6. AntS-v1

            @transfer-learning environments:
                1. Centipede
                2. Snake
                4. Reacher
    '''
    # step 0: settings about the joint
    JOINT_KEY = ['qpos', 'qvel', 'qfrc_constr', 'qfrc_act']
    BODY_KEY = ['cinert', 'cvel', 'cfrc']

    ROOT_OB_SIZE = {
        'qpos': {'free': 7, 'hinge': 1, 'slide': 1},
        'qvel': {'free': 6, 'hinge': 1, 'slide': 1},
        'qfrc_act': {'free': 6, 'hinge': 1, 'slide': 1},
        'qfrc_constr': {'free': 6, 'hinge': 1, 'slide': 1}
    }

    # step 1: register the settings for traditional environments
    SYMMETRY_MAP = {'Humanoid-v1': 2,
                    'HumanoidStandup-v1': 2,
                    'HalfCheetah-v1': 1,
                    'Hopper-v1': 1,
                    'Walker2d-v1': 1,
                    'AntS-v1': 2,
                    'Swimmer-v1': 2,

                    'WalkersHopper-v1': 1,
                    'WalkersHalfhumanoid-v1': 1,
                    'WalkersHalfcheetah-v1': 1,
                    'WalkersFullcheetah-v1': 1,
                    'WalkersOstrich-v1': 1,
                    'WalkersKangaroo-v1': 1}

    XML_DICT = {'Humanoid-v1': 'humanoid.xml',
                'HumanoidStandup-v1': 'humanoid.xml',
                'HalfCheetah-v1': 'half_cheetah.xml',
                'Hopper-v1': 'hopper.xml',
                'Walker2d-v1': 'walker2d.xml',
                'AntS-v1': 'ant.xml',
                'Swimmer-v1': 'SnakeThree.xml',

                'WalkersHopper-v1': 'WalkersHopper.xml',
                'WalkersHalfhumanoid-v1': 'WalkersHalfhumanoid.xml',
                'WalkersHalfcheetah-v1': 'WalkersHalfcheetah.xml',
                'WalkersFullcheetah-v1': 'WalkersFullcheetah.xml',
                'WalkersOstrich-v1': 'WalkersOstrich.xml',
                'WalkersKangaroo-v1': 'WalkersKangaroo.xml'}
    OB_MAP = {
        'Humanoid-v1':
            ['qpos', 'qvel', 'cinert', 'cvel', 'qfrc_act', 'cfrc'],
        'HumanoidStandup-v1':
            ['qpos', 'qvel', 'cinert', 'cvel', 'qfrc_act', 'cfrc'],
        'HalfCheetah-v1': ['qpos', 'qvel'],
        'Hopper-v1': ['qpos', 'qvel'],
        'Walker2d-v1': ['qpos', 'qvel'],
        'AntS-v1': ['qpos', 'qvel', 'cfrc'],
        'Swimmer-v1': ['qpos', 'qvel'],

        'WalkersHopper-v1': ['qpos', 'qvel'],
        'WalkersHalfhumanoid-v1': ['qpos', 'qvel'],
        'WalkersHalfcheetah-v1': ['qpos', 'qvel'],
        'WalkersFullcheetah-v1': ['qpos', 'qvel'],
        'WalkersOstrich-v1': ['qpos', 'qvel'],
        'WalkersKangaroo-v1': ['qpos', 'qvel']
    }

    # step 2: register the settings for the tranfer environments
    SYMMETRY_MAP.update({
        'Centipede': 2,
        'CpCentipede': 2,
        'Snake': 2,
        'Reacher': 0,
    })

    OB_MAP.update({
        'Centipede': ['qpos', 'qvel', 'cfrc'],
        'CpCentipede': ['qpos', 'qvel', 'cfrc'],
        'Snake': ['qpos', 'qvel'],
        'Reacher': ['qpos', 'qvel', 'root_add_5']
    })
    for env in TASK_DICT:
        for i_part in TASK_DICT[env]:
            registered_name = env + num2words.num2words(i_part)[0].upper() \
                + num2words.num2words(i_part)[1:] + '-v1'

            SYMMETRY_MAP[registered_name] = SYMMETRY_MAP[env]
            OB_MAP[registered_name] = OB_MAP[env]
            XML_DICT[registered_name] = registered_name.replace(
                '-v1', '.xml'
            )

    # ob map, symmetry map for robustness task
    for key in ROBUSTNESS_TASK_DICT:
        for env in ROBUSTNESS_TASK_DICT[key]:
            OB_MAP.update({env: ['qpos', 'qvel']})
            SYMMETRY_MAP.update({env: 1})
    # xml dict for botustness task
    for i_agent in range(NUM_ROBUSTNESS_AGENTS + 1):
        XML_DICT.update({
            'WalkersHopper' + num2words.num2words(i_agent) + '-v1':
                'WalkersHopper.xml',
            'WalkersHalfhumanoid' + num2words.num2words(i_agent) + '-v1':
                'WalkersHalfhumanoid.xml',
            'WalkersHalfcheetah' + num2words.num2words(i_agent) + '-v1':
                'WalkersHalfcheetah.xml',
            'WalkersFullcheetah' + num2words.num2words(i_agent) + '-v1':
                'WalkersFullcheetah.xml',
            'WalkersOstrich' + num2words.num2words(i_agent) + '-v1':
                'WalkersOstrich.xml',
        })

    return SYMMETRY_MAP, XML_DICT, OB_MAP, JOINT_KEY, ROOT_OB_SIZE, BODY_KEY
