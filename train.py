import os
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env

env_name = 'ALE/Breakout-v5'
log_path = os.path.join('training', 'logs', 'Breakout')

env = make_atari_env(env_name, n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

a2c_path = os.path.join('training', 'saved_models', 'Breakout', 'A2C_Breakout_6m')
model = A2C.load(a2c_path, env)
# model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)


model.learn(total_timesteps=4000000)

a2c_path = os.path.join('training', 'saved_models', 'Breakout', 'A2C_Asteroids_10m')
model.save(a2c_path)