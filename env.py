import numpy as np
import strategy

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

class PrisonersDilemma:
    def __init__(self, strategy=strategy.Strategy, moves_in_memory=5, max_score=500):
        super().__init__()
        self.max_score = max_score
        self.my_score = self.their_score = 0
        self.move = 0

        self.strategy = strategy()

        # the state space here is last N moves of the strategy
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
        their_action = self.strategy.get_move(action)
            
        payoffs = self.reward_table[action][their_action]
        self.my_score += payoffs[0]
        self.their_score += payoffs[1]
        
        done = True if self.my_score >= self.max_score or self.their_score >= self.max_score else False
        reward = payoffs[0]

        self._update_observation(their_action)
        self.move += 1

        return self._get_obs(), reward, done, None


