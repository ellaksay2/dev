import numpy as np


social_mice = ( 'social-0914-1', 'social-0914-4', 'social-0921-2') #'social_0106_1',


exclude_list = {
}

social_VR_sessions = {
    # 'social_0106_1': (
    #     {'date': '14_03_2024' , 'scene': 'Env1_to_Env2_fixreward', 'session': 1, 'scan':1, 'exp_day':1},
    #     {'date': '14_03_2024' , 'scene': 'soical_dark_background', 'session': 1, 'scan':1, 'exp_day':1},
    #     {'date': '14_03_2024' , 'scene': 'soical_dark_background', 'session': 2, 'scan':1, 'exp_day':1},
    #     {'date': '14_03_2024' , 'scene': 'soical_dark_background', 'session': 3, 'scan':1, 'exp_day':1}
    # ),
    'social-0914-1': (
        # {'date': '02_10_2024' , 'scene': 'soical_dark_background', 'session': 1, 'scan':1, 'exp_day':1},
        # {'date': '02_10_2024' , 'scene': 'soical_dark_background', 'session': 2, 'scan':1, 'exp_day':1},
        # {'date': '02_10_2024' , 'scene': 'soical_dark_background_3', 'session': 2, 'scan':0, 'exp_day':1},
        {'date': '03_10_2024' , 'scene': 'Env1_to_Env2_fixreward', 'session': 1, 'scan':1, 'exp_day':2},
        {'date': '03_10_2024' , 'scene': 'social_emptytunnel', 'session': 1, 'scan':1, 'exp_day':2},
        {'date': '03_10_2024' , 'scene': 'social_restrict_fam', 'session': 1, 'scan':1, 'exp_day':2},
        {'date': '03_10_2024' , 'scene': 'social_unrestrict_fam','session': 1,  'scan':1, 'exp_day':2},
        {'date': '03_10_2024' , 'scene': 'social_wheel_restrict', 'session': 1, 'scan':1, 'exp_day':2},
        {'date': '04_10_2024' , 'scene': 'Env1_to_Env2_fixreward', 'session': 1, 'scan':1, 'exp_day':3},
        {'date': '04_10_2024' , 'scene': 'social_restrict_fam', 'session': 1, 'scan':1, 'exp_day':3},
        {'date': '04_10_2024' , 'scene': 'social_restrict_nov', 'session': 1, 'scan':1, 'exp_day':3},
        # {'date': '04_10_2024' , 'scene': 'social_unrestrict_fam', 'session': 1, 'scan':1, 'exp_day':3},
        {'date': '04_10_2024' , 'scene': 'social_unrestrict_nov', 'session': 1, 'scan':1, 'exp_day':3},
        {'date': '04_10_2024' , 'scene': 'social_wheel_restrict', 'session': 1, 'scan':1, 'exp_day':3},
        {'date': '05_10_2024' , 'scene': 'Env1_to_Env2_fixreward', 'session': 1, 'scan':1, 'exp_day':4},
        {'date': '05_10_2024' , 'scene': 'social_unrestrict_fam', 'session': 1, 'scan':1, 'exp_day':4},
        {'date': '05_10_2024' , 'scene': 'social_unrestrict_nov', 'session': 1, 'scan':1, 'exp_day':4},
        {'date': '06_10_2024' , 'scene': 'Env1_to_Env2_fixreward', 'session': 1, 'scan':1, 'exp_day':5},
        {'date': '06_10_2024' , 'scene': 'social_restrict_nov_diffsex', 'session': 1, 'scan':1, 'exp_day':5},
        # {'date': '06_10_2024' , 'scene': 'social_unrestrict_nov',  'scan':, 'exp_day':5},
        {'date': '06_10_2024' , 'scene': 'social_unrestrict_nov', 'session': 2, 'scan':1, 'exp_day':5},
        {'date': '06_10_2024' , 'scene': 'social_unrestrict_nov_diffsex', 'session': 1, 'scan':1, 'exp_day':5},
        {'date': '06_10_2024' , 'scene': 'social_wheel_restrict', 'session': 1, 'scan':1, 'exp_day':5},
    ),
    'social-0914-4': (
        {'date': '03_10_2024' , 'scene': 'Env1_to_Env2_fixreward', 'session':1, 'scan':1, 'exp_day':2},
        {'date': '03_10_2024' , 'scene': 'social_emptytunnel', 'session': 1, 'scan':1, 'exp_day':1},
        {'date': '03_10_2024' , 'scene': 'social_restrict_fam', 'session': 1, 'scan':1, 'exp_day':1},
        {'date': '03_10_2024' , 'scene': 'social_unrestrict_fam', 'session': 1, 'scan':1, 'exp_day':1},
        {'date': '03_10_2024' , 'scene': 'social_wheel_restrict', 'session': 1, 'scan':1, 'exp_day':1},
        {'date': '04_10_2024' , 'scene': 'Env1_to_Env2_fixreward',  'session': 1,'scan':1, 'exp_day':2},
        {'date': '04_10_2024' , 'scene': 'social_restrict_fam', 'session': 1, 'scan':1, 'exp_day':2},
        {'date': '04_10_2024' , 'scene': 'social_restrict_nov', 'session': 1, 'scan':1, 'exp_day':2},
        {'date': '04_10_2024' , 'scene': 'social_unrestrict_fam', 'session': 1, 'scan':1, 'exp_day':2},
        {'date': '04_10_2024' , 'scene': 'social_unrestrict_nov', 'session': 1, 'scan':1, 'exp_day':2},
        {'date': '04_10_2024' , 'scene': 'social_wheel_restrict', 'session': 1, 'scan':1, 'exp_day':2},
        {'date': '05_10_2024' , 'scene': 'Env1_to_Env2_fixreward',  'session': 1,'scan':1, 'exp_day':3},
        # {'date': '05_10_2024' , 'scene': 'social_unrestrict_fam',  'scan':, 'exp_day':3},
        {'date': '05_10_2024' , 'scene': 'social_unrestrict_fam', 'session': 1, 'scan':1, 'exp_day':3},
        {'date': '05_10_2024' , 'scene': 'social_unrestrict_nov', 'session': 1, 'scan':1, 'exp_day':3},
        {'date': '06_10_2024' , 'scene': 'Env1_to_Env2_fixreward', 'session': 1, 'scan':1, 'exp_day':4},
        {'date': '06_10_2024' , 'scene': 'social_restrict_nov_diffsex', 'session': 1, 'scan':1, 'exp_day':4},
        {'date': '06_10_2024' , 'scene': 'social_unrestrict_nov', 'session': 1, 'scan':1, 'exp_day':4},
        # {'date': '06_10_2024' , 'scene': 'social_unrestrict_nov',  'scan':, 'exp_day':4},
        {'date': '06_10_2024' , 'scene': 'social_unrestrict_nov_diffsex','session': 1,  'scan':1, 'exp_day':4},
        {'date': '06_10_2024' , 'scene': 'social_wheel_restrict', 'session': 1, 'scan':1, 'exp_day':4},
    ),
    'social-0921-2': (
        # {'date': '03_10_2024' , 'scene': 'Env1_to_Env2_fixreward',  'scan':, 'exp_day':2},
        {'date': '03_10_2024' , 'scene': 'social_emptytunnel','session': 1,  'scan':1, 'exp_day':1},
        {'date': '03_10_2024' , 'scene': 'social_restrict_fam', 'session': 1, 'scan':1, 'exp_day':1},
        {'date': '03_10_2024' , 'scene': 'social_unrestrict_fam', 'session': 1, 'scan':1, 'exp_day':1},
        {'date': '03_10_2024' , 'scene': 'social_wheel_restrict', 'session': 1, 'scan':1, 'exp_day':1},
        # {'date': '04_10_2024' , 'scene': 'Env1_to_Env2_fixreward',  'scan':, 'exp_day':2},
        {'date': '04_10_2024' , 'scene': 'social_restrict_fam', 'session': 1, 'scan':1, 'exp_day':2},
        {'date': '04_10_2024' , 'scene': 'social_restrict_nov', 'session': 1, 'scan':1, 'exp_day':2},
        {'date': '04_10_2024' , 'scene': 'social_unrestrict_fam', 'session': 1, 'scan':1, 'exp_day':2},
        {'date': '04_10_2024' , 'scene': 'social_unrestrict_nov', 'session': 1, 'scan':1, 'exp_day':2},
        {'date': '04_10_2024' , 'scene': 'social_wheel_restrict', 'session': 1, 'scan':1, 'exp_day':2},
        # {'date': '05_10_2024' , 'scene': 'Env1_to_Env2_fixreward',  'scan':, 'exp_day':3},
        {'date': '05_10_2024' , 'scene': 'social_unrestrict_fam', 'session': 1, 'scan':1, 'exp_day':3},
        # {'date': '05_10_2024' , 'scene': 'social_unrestrict_fam',  'scan':, 'exp_day':3},
        {'date': '05_10_2024' , 'scene': 'social_unrestrict_nov', 'session': 1, 'scan':1, 'exp_day':3},
        # {'date': '06_10_2024' , 'scene': 'Env1_to_Env2_fixreward',  'scan':, 'exp_day':4},
        {'date': '06_10_2024' , 'scene': 'social_restrict_nov_diffsex', 'session': 1, 'scan':1, 'exp_day':4},
        {'date': '06_10_2024' , 'scene': 'social_unrestrict_nov', 'session': 1, 'scan':1, 'exp_day':4},
        # {'date': '06_10_2024' , 'scene': 'social_unrestrict_nov',  'scan':, 'exp_day':4},
        {'date': '06_10_2024' , 'scene': 'social_unrestrict_nov_diffsex', 'session': 1, 'scan':1, 'exp_day':4},
        {'date': '06_10_2024' , 'scene': 'social_wheel_restrict', 'session': 1, 'scan':1, 'exp_day':4},
    )
}

social_SLP_sessions = {
    'social_0106': (
        
    )
}
        

