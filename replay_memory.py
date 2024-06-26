import random
from collections import namedtuple, deque
import torch

######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classes:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

#Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
Transition = namedtuple('Transition', ('state', 'reward', 'next_state', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def shuffle(self):
        """Shuffles the entire memory."""
        # Convert deque to a list, shuffle, and convert back to deque
        temp_list = list(self.memory)
        random.shuffle(temp_list)
        self.memory = deque(temp_list, maxlen=self.memory.maxlen)

    def __len__(self):
        return len(self.memory)