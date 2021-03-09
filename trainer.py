import numpy as np
from data_loader import *
import torch
import matplotlib.pyplot as plt

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
        macd_val, sig_val = compute_macd(self._stock_dataset.val_df)
        
        self._train_data = torch.tensor(np.vstack((macd, sig, self._stock_dataset.train_df["Adj Close"])).T)
        self._val_data = torch.tensor(np.vstack((macd_val, sig_val, self._stock_dataset.val_df["Adj Close"])).T)
        
        #train_windows = torch.tensor(window_data(self._train_data, self.args.lstm_timesteps))


        # Convert to Tensor and add batch dimension
        #state = torch.tensor(init_state, dtype=torch.float32).cuda().unsqueeze(1)
        
        # print(init_state.shape)
        losses = []
        final_balance = []
        final_actions = []
        for ep_i in range(self.args.n_episodes):
            self.reset()
            
            # Set number of days to train
            if not self.args.no_rand_days:
                num_train_days = np.random.randint(self.args.min_train_days, self.args.max_train_days+1) 
            else:
                num_train_days = self.args.max_train_days

            # Set start day
            if not self.args.no_rand_start:
                init_i = np.random.randint(0, len(self._train_data) - num_train_days + 1)
            else:
                init_i = 0

            if ep_i + 1 == self.args.n_episodes:
                num_train_days = self.args.max_train_days
                #init_i = 0#len(self._train_data) - num_train_days
                self.reset()
                self._balance = torch.tensor([self.args.max_init_balance])
                self._num_stock = torch.zeros(1)
            
            state = self.setup_init_state(init_i=init_i)
            for i in range(num_train_days):
                #print(state)
                #cur_close_idx = i + self.args.lstm_timesteps - 1
                action, q_value = self._agent_trader(state, argmax=ep_i + 1 == self.args.n_episodes)

                # Convert action to [-max_trans, max_trans]
                num_trans = action - self.args.max_trans
                cur_close_price = self._unstandardize_price(self._train_data[init_i+i][-1])
                pre_trans_balance = self._balance.clone()
                print("q_value", q_value)
                print("BEFORE", num_trans, cur_close_price, self._num_stock, self._balance)
                self._transaction(num_trans, cur_close_price)
                print("AFTER", num_trans, cur_close_price, self._num_stock, self._balance, "\n")

                # Create next state
                next_state = state.clone()
                next_state = next_state.roll(-1, 1) # Move states down
                next_state[:, -1] = torch.hstack((self._train_data[init_i+i], torch.log(self._balance+1), torch.log(self._num_stock+1))) 
                # else:
                #     # There is not a next state
                #     next_state = None
                
                # Create experience
                #print(pre_trans_balance, self._balance)
                reward = torch.log(self._balance / pre_trans_balance)
                e_t = Experience(state, action, reward, next_state)
                self._agent_trader.add_ex(e_t)

                if ep_i +1 == self.args.n_episodes:
                    final_balance.append(self._balance.numpy()[0])
                    final_actions.append(action)

                # Update state
                state = next_state

                if (i+1) % self.args.batch_size == 0 and ep_i + 1 < self.args.n_episodes:
                    print("TRAINING")
                    loss = self._agent_trader.train()
                    losses.append(loss)
            
            # Update the target network
            self._agent_trader.update_target()
        
        # Sell all the rest of your stock
        cur_close_price = self._unstandardize_price(self._train_data[init_i + i][-1])
        print(self._train_data[i][-1], cur_close_price, self._num_stock)
        self._transaction(-self._num_stock, cur_close_price)
        final_balance.append(self._balance.numpy()[0])
        print(final_balance)
        
        fig, axs = plt.subplots(3)
        axs[0].plot(losses)
        axs[0].set_title("L1 Loss")

        axs[1].plot(final_balance)
        axs[1].set_title("Final Episode Balance")
        axs[2].plot(self._train_data[init_i:init_i+num_train_days, -1])
        axs[2].set_title("Price")
        
        plt.show()

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
        if not self.args.no_rand_balance:
            init_balance = float(np.random.randint(self.args.min_init_balance, self.args.max_init_balance))
        
        self._balance = torch.tensor([init_balance])
        self._num_stock = torch.zeros(1)

    
    def setup_init_state(self, init_i=0):
        balance = torch.repeat_interleave(torch.log(self._balance), self.args.lstm_timesteps).reshape(self.args.lstm_timesteps, 1)
        cur_stock = torch.zeros((self.args.lstm_timesteps,1))
        stock_data = torch.zeros((self.args.lstm_timesteps, 3))
        stock_data[-1] = self._train_data[init_i]
        state = torch.hstack(
            (stock_data, balance, cur_stock)).cuda().float().unsqueeze(0)

        cur_stock = torch.zeros((self.args.lstm_timesteps,1))
        stock_data = torch.zeros((self.args.lstm_timesteps, 3))
        state = torch.hstack((stock_data, balance, cur_stock)).cuda().float().unsqueeze(0)
        return state