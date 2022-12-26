sampling_policy = {
    'motiv_medium': ['medium_expert'] * 5 + ['medium_replay'] * 5,
    'motiv_low': ['medium_expert'] * 3 + ['medium_replay'] * 7,
    'motiv_high': ['medium_expert'] * 7 + ['medium_replay'] * 3,
    'mixed_low': ['medium_expert'] * 2 + ['replay'] * 3 + ['medium_replay'] * 5,
    'mixed_medium': ['medium_expert' ] * 3 + ['replay'] * 3 + ['medium_replay'] * 4,
    'mixed_high': ['medium_expert'] * 4 + ['replay'] * 3 + ['medium_replay'] * 3,

    'expert_low': ['medium_expert'] * 3 + ['replay'] * 7,
    'expert_medium': ['medium_expert'] * 5 + ['replay'] * 5,
    'medium_replay': ['medium_replay'] * 10,
    'replay': ['replay'] * 10,
    'medium_expert': ['medium_expert'] * 10,

    'airsim_medium_expert': ['medium_expert'] * 6,
    'airsim_medium_replay': ['medium_replay'] * 6,
    'airsim_replay': ['replay'] * 6,
    'airsim_low': ['medium_replay'] * 3 + ['replay'] * 2 + ['medium_expert'],
    'airsim_medium': ['medium_replay'] * 2 + ['replay'] * 2 + ['medium_expert'] * 2,
    'airsim_high': ['medium_replay'] + ['replay'] * 2 + ['medium_expert'] * 3,

}


default_dynamic_num_data = {'medium_replay': 150, 'replay': 100, 'medium_expert': 50}