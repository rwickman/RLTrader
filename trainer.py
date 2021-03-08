import numpy as np
from models import Trader
from data_loader import *
import torch

class Trainer:
    def __init__(self, args):
        self.args = args
        self._agent_trader = Trader(self.args).cuda()
        
    def train(self):
        train_df, val_df, test_df = load_data(self.args.data_csv)
        macd, sig = compute_macd(train_df)
        
        train_data = np.vstack((macd, sig, train_df["Adj Close"])).T
        train_windows = window_data(train_data, self.args.lstm_timesteps)
        
        init_state = np.hstack(
            (train_windows[0], np.zeros((self.args.lstm_timesteps,1)), np.zeros((self.args.lstm_timesteps,1))))
        
        # Convert to Tensor and add batch dimension
        init_state = torch.tensor(init_state, dtype=torch.float32).cuda().unsqueeze(1)
        print(init_state.shape)
        action = self._agent_trader(init_state)
        print(action)