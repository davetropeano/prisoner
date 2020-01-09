import numpy as np
import time, pickle, os

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

    def get_move(self):
        eps = np.random.uniform()
        move = 1 if eps > 0.5 else 0
        self.moves.append(move)
        return move

class PrisonersDilemma:
    def __init__(self, moves_in_memory=5, max_score=500):
        super().__init__()
        self.max_score = max_score
        self.me_score = self.you_score = 0
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
        self.my_score = self.you_score = 0
        self.move = 0
        self.observation_space.reset()
        return self._get_obs()

    def render(self):
        print('---------------------')
        print('me: %s, you: %s, memory: %s' % (self.me_score, self.you_score, self._get_obs()))

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
        you_action = self.opponent.get_move()
        payoffs = self.reward_table[action][you_action]
        self.me_score += payoffs[0]
        self.you_score += payoffs[1]
        
        done = True if self.me_score >= self.max_score or self.you_score >= self.max_score else False
        reward = payoffs[0]

        self._update_observation(you_action)
        self.move += 1

        return self._get_obs(), reward, done, None


env = PrisonersDilemma()

epsilon = 0.9
# min_epsilon = 0.1
# max_epsilon = 1.0
# decay_rate = 0.01

total_episodes = 10000

lr_rate = 0.81
gamma = 0.96

Q = np.zeros((2**env.observation_space.n, env.action_space.n))
    
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
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
        t = 0
        reward_total = 0

        obs = env.reset()
        state = observation_to_state(obs)
        action = choose_action(state)

        # Setting max steps per game (episode) to a constant drives to a defect-only strategy
        # by induction. Instead, don't let the AI train only a fixed number of steps and instead vary this
        max_steps = np.random.randint(100, 1000)    


        done = False
        while not done:
            obs, reward, done, info = env.step(action)
            state2 = observation_to_state(obs)

            action2 = choose_action(state2)

            learn(state, state2, reward, action, action2)

            state = state2
            action = action2

            t += 1
            reward_total += reward

        rewards.append(reward_total)
        env.render()

    # epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode) 
    # os.system('clear')

        
    print ("Score over time: ", sum(rewards)/total_episodes)
    print(Q)

    with open("pd_qTable_sarsa.pkl", 'wb') as f:
        pickle.dump(Q, f)