import gym
import os
from stable_baselines3 import A2C
import time
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

env_name = 'ALE/Breakout-v5'
env = make_atari_env(env_name, n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)
a2c_path = os.path.join('training', 'saved_models', 'Breakout', 'A2C_Breakout_25m')
model = A2C.load(a2c_path, env)

observation = env.reset()
score = 0
done = [False, False, False, False]
denzel = False
# episodes = 5
# for eposide in range(1, episodes):
while not denzel:
    action, _ = model.predict(observation)
    observation, reward, done, info = env.step(action)
    score += reward
    time.sleep(0.3)
    env.render(mode='human')
env.close()