import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

from replay_memory import ReplayMemory
from replay_memory import PrioritizedExpReplay
# class Critic(nn.Module)

class DDQNAgent:
    def __init__(self, args):
        self.args = args
        self._dqn = DQN(self.args).cuda()
        self._dqn_target = DQN(self.args).cuda()
        
        #self._replay_memory = ReplayMemory(self.args)
        self._replay_memory = PrioritizedExpReplay(self.args)

        self._optimizer = optim.Adam(self._dqn.parameters(), lr=self.args.lr)
        self._loss_fn = nn.SmoothL1Loss()
        self._num_steps = 0
        self._action_dim = self.args.max_trans * 2 + 1
        self._model_file = os.path.join(self.args.save_dir, "dqn_model.pt")
        if not os.path.exists("models"):
            os.mkdir("models")
        if self.args.load:
            self.load()

    def __call__(self, x, argmax=False):
        return self.get_action(self._dqn(x), argmax)

    def get_action(self, q_values, argmax=False):
        q_values = torch.squeeze(q_values)
        if not argmax and self.args.epsilon >= np.random.rand():
            # Perform random action
            action = np.random.randint(self._action_dim)
        else:
            with torch.no_grad():
                # Perform action that maximizes expected return
                action =  q_values.max(0)[1]
        action = int(action)
        return action, q_values[action]

    def add_ex(self, e_t):
        """Add a step of experience."""
        with torch.no_grad():
            q_value = self._dqn(e_t.state)[0][e_t.action]
            if e_t.next_state is not None:
                next_action, _ = self.get_action(self._dqn(e_t.next_state))
                q_next_target = self._dqn_target(e_t.next_state)[0, next_action]
                td_target = e_t.reward[0].cuda() + self.args.gamma *  q_next_target
            else:
                td_target = e_t.reward
            
            td_error = td_target - q_value 

        self._replay_memory.add(e_t, td_error)

    def train(self):
        is_ws, exs, indices = self._replay_memory.sample(self.args.batch_size)
        td_targets = torch.zeros(self.args.batch_size).cuda()
        states = torch.zeros(self.args.batch_size, self.args.lstm_timesteps, self.args.state_size).cuda()
        next_states = torch.zeros(self.args.batch_size, self.args.lstm_timesteps, self.args.state_size).cuda()
        rewards = torch.zeros(self.args.batch_size)
        next_state_mask = torch.zeros(self.args.batch_size)
        actions = []

        # Create state-action values
        for i, e_t in enumerate(exs):
            states[i] = e_t.state
            actions.append(e_t.action)
            rewards[i] = e_t.reward
            
            # Check if next state exists
            if e_t.next_state is not None:
                next_states[i] = e_t.next_state 
                next_state_mask[i] = 1
        
        # Select the q-value for every state
        actions = torch.tensor(actions, dtype=torch.int64).cuda()
        q_values = self._dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Create TD targets
        q_next  = self._dqn(next_states)
        q_next_target  = self._dqn_target(next_states)
        for i in range(self.args.batch_size):
            if next_state_mask[i] == 0:
                td_targets[i] = rewards[i]
            else:
                # Get the argmax next action for DQN
                action, _ = self.get_action(q_next[i], True)
                
                # Set TD Target using the q-value of the target network
                # This is the Double-DQN target
                td_targets[i] = rewards[i] + self.args.gamma * q_next_target[i, action]

        #print("rewards", rewards, "SUM REWARDS", sum(rewards))
        #print("q_values", q_values)
        #print("td_targets", td_targets)
        # Train model
        self._optimizer.zero_grad()        
        td_errors = q_values - td_targets
        loss = torch.mean(td_errors ** 2  *  is_ws) #self._loss_fn(q_values[0], td_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._dqn.parameters(), self.args.grad_clip)
        #print("GRAD MAX", self._dqn.q_values.weight.grad.max())
        print("loss", loss)
        self._optimizer.step()
        self._num_steps += 1
        self._replay_memory.update_priorities(indices, td_errors.detach())
        return loss

    def update_target(self):
        self._dqn_target.load_state_dict(self._dqn.state_dict())

    def replay_len(self):
        return self._replay_memory.current_capacity()

    def save(self):
        model_dict = {
            "DQN" : self._dqn.state_dict(),
            "optimizer" : self._optimizer.state_dict() 
        }
        

        torch.save(model_dict, self._model_file)

    def load(self):
        model_dict = torch.load(self._model_file)
        self._dqn.load_state_dict(model_dict["DQN"])
        self._dqn_target.load_state_dict(model_dict["DQN"])
        self._optimizer.load_state_dict(model_dict["optimizer"])
        
        #self._dqn.load_state_dict(torch.load(self._model_file))
        #self._dqn_target.load_state_dict(torch.load(self._model_file))

class DQN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.state_encoder_fc1 = nn.Linear(self.args.state_size, self.args.hidden_size)
        self.state_encoder_fc2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        self.lstm_cell = nn.LSTMCell(self.args.hidden_size, self.args.hidden_size)

        # Action space is [-max_trans, max_trans]
        self.q_values = nn.Linear(self.args.hidden_size, self.args.max_trans * 2 + 1)

        #self.critic_out = nn.Linear(self.args.hidden_size, 1)

    def forward(self, states):
        batch_size = states.shape[0]
        # Hidden state and cell memory
        
        h_t = torch.zeros(batch_size, self.args.hidden_size).cuda()
        c_t = torch.zeros(batch_size, self.args.hidden_size).cuda()
        for i in range(self.args.lstm_timesteps):
            enc_state = F.relu(self.state_encoder_fc1(states[:, i]))
            enc_state = F.relu(self.state_encoder_fc2(enc_state))

            h_t, c_t = self.lstm_cell(enc_state, (h_t, c_t))

        # Return logits over actions
        return self.q_values(h_t)
