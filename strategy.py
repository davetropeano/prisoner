import numpy as np

class Strategy:
    def __init__(self):
        self.moves = []
        self.last_move = None

    def get_move(self, their_move=None):
        move = np.random.randint(0,2) if self.last_move != None else 0
        self.last_move = move
        self.moves.append(move)
        return move


class Cooperate(Strategy):
    def get_move(self, their_move=None):
        move = 0
        self.last_move = move
        self.moves.append(move)
        return move

class Defect(Strategy):
    def get_move(self, their_move=None):
        move = 1
        self.last_move = move
        self.moves.append(move)
        return move

class Alternating(Strategy):
    def get_move(self, their_move=None):
        move = 0 if self.last_move == None or self.last_move == 1 else 1
        self.last_move = move
        self.moves.append(move)
        return move

class TitForTat(Strategy):
    def __init__(self):
        super().__init__()
        self.memory = None

    def get_move(self, their_move=None):
        move = 0 if self.memory == None else their_move
        self.memory = their_move
        
        self.last_move = move
        self.moves.append(move)
        return move

class _PatternStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.pattern = []
        self.index = 0

class CCD(_PatternStrategy):
    def __init__(self):
        super().__init__()
        self.pattern = [0,0,1]

    def get_move(self, their_move=None):
        move = self.pattern[self.index]
        self.index = (self.index + 1) % len(self.pattern)
        self.last_move = move
        self.moves.append(move)
        return move

class DDC(_PatternStrategy):
    def __init__(self):
        super().__init__()
        self.pattern = [1,1,0]

    def get_move(self, their_move=None):
        move = self.pattern[self.index]
        self.index = (self.index + 1) % len(self.pattern)
        self.last_move = move
        self.moves.append(move)
        return move
