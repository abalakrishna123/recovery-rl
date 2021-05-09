import gym
from gym.envs.registration import register

ENV_ID = {
    'navigation1': 'Navigation-v0',
    'navigation2': 'Navigation-v1',
    'maze': 'Maze-v0',
    'image_maze': 'ImageMaze-v0',
    'obj_extraction': 'ObjExtraction-v0',
    'obj_dynamic_extraction': 'ObjDynamicExtraction-v0',
}

ENV_CLASS = {
    'navigation1': 'Navigation1',
    'navigation2': 'Navigation2',
    'maze': 'MazeNavigation',
    'image_maze': 'MazeImageNavigation',
    'obj_extraction': 'ObjExtraction',
    'obj_dynamic_extraction': 'ObjDynamicExtraction',
}


def register_env(env_name):
    env_id = ENV_ID[env_name]
    env_class = ENV_CLASS[env_name]
    register(id=env_id, entry_point='env.' + env_name + ":" + env_class)


def make_env(env_name):
    env_id = ENV_ID[env_name]
    return gym.make(env_id)
