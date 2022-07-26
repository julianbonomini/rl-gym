import gym

env = gym.make('ALE/SpaceInvaders-v5', render_mode="human")
height, width, channels = env.observation_space.shape
actions = env.action_space.n


episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render(mode='rgb_array')
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()