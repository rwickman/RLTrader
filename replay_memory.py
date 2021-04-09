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

class PrioritizedExpReplay:
    def __init__(self, args):
        self.args = args
        self._sum_tree = SumTree(self.args)
        
    def append(self, e_t):
        """Append experience."""
        self._sum_tree.add(e_t)

    def sample(self):
        """Sample batch size experience replay."""
        
        return np.random.choice(self._memory, size=self.args.batch_size, replace=False)

    def current_capacity(self):
        return self._sum_tree.current_capacity()


class SumTree:
    def __init__(self, args):
        self.args = args
        # sum tree 
        self.tree = torch.zeros(2 * self.args.capacity - 1)
        self.memory = []
        # Pointer to end of memory
        self._end_pos = 0
    
    def add(self, e_t, priority):
        """Add experience to sum tree."""
        
        # Add experience to memory
        if len(self.memory) < self.args.capacity:
            self.memory.append(e_t)
        else:
            self.memory[self._end_pos] = e_t
        
        idx = self.args.capacity + self._end_pos - 1
    
        # Update memorysum tree
        self.update(idx, priority)
        
        # Update end pointer
        self._end_pos = (self._end_pos + 1) % self.args.capacity

    def update(self, idx, priority):
        """Update priority of element and propagate through tree."""
        # Compute priority difference
        diff = priority - self.tree[idx]

        # Propagate update through tree
        while idx >= 0:
            self.tree[idx] += diff
            # Update to parent idx
            idx = (idx - 1) // 2

    def get(self, val):
        """Sample from sum tree based on the sampled value."""
        tree_idx = self._retrieve(0, val)
        data_idx = tree_idx - self.args.capacity + 1

        return self.tree[tree_idx], self.memory[data_idx]

    def _retrieve(self, idx, val):
        left = 2 * idx + 1
        right = 2 * idx + 2
        while left < len(self.tree):
            if val <= self.tree[left]:
                idx = left
            else:
                idx = right
                val -= self.tree[left]

            left = 2 * idx + 1
            right= 2 * idx + 2

        return idx

    def _current_capacity(self):
        return len(self.memory)