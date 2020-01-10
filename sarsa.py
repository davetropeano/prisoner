import numpy as np
import time, pickle, os
import matplotlib.pyplot as pyplot

class ObservationSpace:
    def __init__(self, n):
        self.n = n
        self.space = np.zeros(self.n)

    def reset(self):
        self.space = np.zeros(self.n)

    def update_space(self, space):
        self.space = space

class ActionSpace:
    def __init__(self, n):
        self.n = n
    
    def sample(self):
        return np.random.randint(0, self.n)

class Opponent:
    def __init__(self):
        self.moves = []
        self.last_move = 1

    def get_move(self):
        """
        eps = np.random.uniform()
        move = 1 if eps > 0.5 else 0
        """

        # alternate
        move = 0 if self.last_move == 1 else 1
        self.last_move = move
        self.moves.append(move)
        return move

class PrisonersDilemma:
    def __init__(self, moves_in_memory=5, max_score=500):
        super().__init__()
        self.max_score = max_score
        self.my_score = self.their_score = 0
        self.move = 0

        self.opponent = Opponent()

        # the state space here is last N moves of the opponent
        self.observation_space = ObservationSpace(moves_in_memory)
        self.action_space = ActionSpace(2) # actions are either cooperate (0) or defect (1)

        """
        Use the reward table from: http://web.stanford.edu/class/psych209/Readings/2017ProjectExamples/wangkeven_17581_1628229_psych209_paper.pdf
        where T>R>P>S and
        T=5
        R=3
        P=1
        S=0

        reward_table is structured as

               C        D
        C   me/you   me/you
        D   me/you   me/you

        where "me" and "you" are respective scores based on the decision made
        """
        self.reward_table = [
            [[3,3], [0,5]],
            [[5,0], [1,1]]
        ]

    def reset(self):
        self.my_score = self.their_score = 0
        self.move = 0
        self.observation_space.reset()
        return self._get_obs()

    def render(self):
        print('me: %s, them: %s, memory: %s' % (self.my_score, self.their_score, self._get_obs()))

    def _get_obs(self):
        return self.observation_space.space
    
    def _update_observation(self, action):
        old_state = self._get_obs()
        new_state = np.zeros(self.observation_space.n)

        i = 0
        while i < self.observation_space.n-1:
            new_state[i+1] = old_state[i]
            i += 1
        new_state[0] = action
        self.observation_space.update_space(new_state)

    def step(self, action):
        # assume always an allowed action
        their_action = self.opponent.get_move()
        payoffs = self.reward_table[action][their_action]
        self.my_score += payoffs[0]
        self.their_score += payoffs[1]
        
        done = True if self.my_score >= self.max_score or self.their_score >= self.max_score else False
        reward = payoffs[0]

        self._update_observation(their_action)
        self.move += 1

        return self._get_obs(), reward, done, None


env = PrisonersDilemma()

epsilon = 1.0
total_episodes = 100
max_steps = 300

lr_rate = 0.81
gamma = 0.96

Q = np.zeros((2**env.observation_space.n, env.action_space.n))
    
def choose_action(state):
    action=0
    if np.random.random() >= 1-epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action, action2):
    predict = Q[state, action]
    target = reward + gamma * Q[state2, action2]
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

def observation_to_state(obs):
    state = 0
    for i in range(len(obs)):
        state = 2*state + obs[i]
    return int(state)

# Start
if __name__ == "__main__":
    rewards = []

    for episode in range(total_episodes):
        print(epsilon)
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
            if t == max_steps:
                done = True

        rewards.append(reward_total)
        epsilon -= 2/(total_episodes) if epsilon > 0 else 0

        #env.render()

    print(rewards)
    print ("Score over time: ", sum(rewards)/total_episodes)
    # print(Q)
    pyplot.plot(rewards, 'b--')

    with open("pd_qTable_sarsa.pkl", 'wb') as f:
        pickle.dump(Q, f)