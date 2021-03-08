import numpy as np
from data_loader import *
import torch

from models import DDQNAgent
from replay_memory import Experience

class Trainer:
    def __init__(self, args):
        self.args = args
        self._agent_trader = DDQNAgent(self.args)        
        #self._balance = torch.tensor([self.args.init_balance])
        #self._num_stock = torch.zeros(1)
        self._stock_dataset = load_data(self.args.data_csv)
        self.reset()

    def train(self):
        self._stock_dataset = load_data(self.args.data_csv)
        macd, sig = compute_macd(self._stock_dataset.train_df)
        
        train_data = torch.tensor(np.vstack((macd, sig, self._stock_dataset.train_df["Adj Close"])).T)
        #train_windows = torch.tensor(window_data(train_data, self.args.lstm_timesteps))

        balance = torch.repeat_interleave(self._balance, self.args.lstm_timesteps).reshape(self.args.lstm_timesteps, 1)
        cur_stock = torch.zeros((self.args.lstm_timesteps,1))
        stock_data = torch.zeros((self.args.lstm_timesteps, 3))
        stock_data[-1] = train_data[0]
        #print(train_data.shape)
        #print(train_data[:self.args.lstm_timesteps].shape)
        state = torch.hstack(
            (stock_data, balance, cur_stock)).cuda().float().unsqueeze(0)

        # Convert to Tensor and add batch dimension
        #state = torch.tensor(init_state, dtype=torch.float32).cuda().unsqueeze(1)
        
        # print(init_state.shape)

        for episode in range(self.args.n_episodes):
            self.reset()
            state = torch.hstack((stock_data, balance, cur_stock)).cuda().float().unsqueeze(0)
            for i in range(64):
                #print(state)
                #cur_close_idx = i + self.args.lstm_timesteps - 1
                action, q_value = self._agent_trader(state)

                # Convert action to [-max_trans, max_trans]
                num_trans = action - self.args.max_trans

                cur_close_price = self._unstandardize_price(train_data[i][-1])
                pre_trans_balance = self._balance.clone()
                print("q_value", q_value)
                print("BEFORE", num_trans, cur_close_price, self._num_stock, self._balance)
                self._transaction(num_trans, cur_close_price)
                print("AFTER", num_trans, cur_close_price, self._num_stock, self._balance, "\n")

                # Create next state
                if i < 255:
                    next_state = state.clone()
                    next_state = next_state.roll(-1, 1) # Move states down
                    next_state[:, -1] = torch.hstack((train_data[i+1], self._balance, self._num_stock)) 
                else:
                    next_state = None
                
                # Create experience
                #print(pre_trans_balance, self._balance)
                reward = torch.log(pre_trans_balance / self._balance)
                e_t = Experience(state, action, reward, next_state)
                self._agent_trader.add_ex(e_t)

                #print(q_value, reward)

                # Update state
                state = next_state

                if (i+1) % self.args.batch_size == 0:
                    print("TRAINING")
                    self._agent_trader.train()

            # Update the target network
            self._agent_trader.update_target()
        
        # Save DQN model
        self._agent_trader.save()

    def _transaction(self, num_trans, cur_close_price):
        """Handle buy/sell transaction."""
        
        # Sell
        if num_trans < 0:
            
            # Can only sell a maximum of stock owned
            cur_sell = min(self._num_stock, -num_trans)
            profit = cur_sell * cur_close_price

            # Update stock and balance
            self._balance += profit
            self._num_stock -= cur_sell
        # Buy
        elif num_trans > 0:
            # Can only buy a maximum based on account balance
            max_purchase = self._balance//cur_close_price
            print("max_purchase", max_purchase) 
            cur_buy = min(max_purchase, num_trans)
            cost_of_purchase = cur_buy * cur_close_price

            # Update stock and balance
            self._balance -= cost_of_purchase
            self._num_stock += cur_buy
        

    
    def _unstandardize_price(self, price):
        return price * self._stock_dataset.train_std + self._stock_dataset.train_mean

    def reset(self):
        self._balance = torch.tensor([self.args.init_balance])
        self._num_stock = torch.zeros(1)