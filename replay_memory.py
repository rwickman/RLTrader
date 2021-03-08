import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor = None

class ReplayMemory:
    def __init__(self, args):
        self.args = args
        self._memory = []
        # Pointer to end of memory
        self._cur_pos = 0

    def append(self, e_t):
        """Append experience."""
        if len(self._memory) >= self.args.capacity:
            self._memory[self._cur_pos] = e_t
        else:
            self._memory.append(e_t)

        # Update end of memory
        self._cur_pos = (self._cur_pos + 1) %  self.args.capacity 

    def sample(self):
        """Sample batch size experience replay."""
        return np.random.choice(self._memory, size=self.args.batch_size, replace=False)

    def current_capacity(self):
        return len(self._memory)