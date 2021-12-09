import gym
import time
from train_acrobot import Agent

if __name__ == '__main__':

    env = gym.make('Acrobot-v1')

    agent = Agent(env)
    agent.load_current_state("model/model.pth")

    state = env.reset()
    score = 0
    done = False

    while not done:
        next_state, reward, done = agent.step(state, 0)

        score += reward
        state = next_state

        env.render()
        time.sleep(0.03)