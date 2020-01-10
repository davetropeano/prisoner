import numpy as np
import time, pickle, os
import matplotlib.pyplot as pyplot
from env import PrisonersDilemma
import strategy

env = PrisonersDilemma(strategy=strategy.DDC)

EPS = 1.0
TOTAL_EPISODES = 100
MAX_STEPS = 50

LEARNING_RATE = 0.25 #0.81
GAMMA = 0.90 # 0.96

# since 0 means cooperate we are initializing the Q table to always cooperate
# unless a state has been visited enough times to make defect the more dominant strategy
# for that state

Q = np.zeros((2**env.observation_space.n, env.action_space.n))
    
def choose_action(state):
    action = np.argmax(Q[state]) if np.random.random() < 1-EPS else env.action_space.sample()
    return action

def learn(state, state2, reward, action, action2):
    predict = Q[state, action]
    target = reward + GAMMA * Q[state2, action2]
    Q[state, action] = Q[state, action] + LEARNING_RATE * (target - predict)

def observation_to_state(obs):
    state = 0
    for i in range(len(obs)):
        state = 2*state + obs[i]
    return int(state)

# Start
if __name__ == "__main__":
    rewards = []

    for episode in range(TOTAL_EPISODES):
        t = 0
        reward_total = 0

        obs = env.reset()
        state = observation_to_state(obs)
        action = choose_action(state)

        done = False
        while not done:
            obs, reward, done, info = env.step(action)
            state2 = observation_to_state(obs)
            reward_total += reward

            action2 = choose_action(state2)

            learn(state, state2, reward, action, action2)

            state = state2
            action = action2

            t += 1
            if t == MAX_STEPS:
                done = True

        rewards.append(reward_total)
        EPS -= 2/(TOTAL_EPISODES) if EPS > 0.02 else 0

        #env.render()

    print ("Score over time: ", sum(rewards)/TOTAL_EPISODES)
    print(Q)
    pyplot.plot(rewards, 'b--')
    pyplot.show()

    with open("pd_qTable_sarsa.pkl", 'wb') as f:
        pickle.dump(Q, f)