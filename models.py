import torch
import torch.nn as nn

# class Critic(nn.Module)

class Trader(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.state_size = 5 

        self.lstm_cell = nn.LSTMCell(self.state_size, self.args.hidden_size)

        # Action space is [-max_trans, max_trans]
        self.agent_out = nn.Linear(self.args.hidden_size, self.args.max_trans * 2 + 1)

    def forward(self, states):
        # Hidden state and cell memory
        h_t = torch.zeros(1, self.args.hidden_size).cuda()
        c_t = torch.zeros(1, self.args.hidden_size).cuda()
        for i in range(self.args.lstm_timesteps):
            h_t, c_t = self.lstm_cell(states[i], (h_t, c_t))
        
        # Return logits over actions
        return self.agent_out(h_t)