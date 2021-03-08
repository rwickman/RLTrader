import argparse
from trainer import Trainer
def main(args):
    args.state_size = 5
    model_trainer = Trainer(args)
    model_trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true",
        help="Load saved models.")
    parser.add_argument("--save_dir", default="models",
        help="Directory to save the models.")
    parser.add_argument("--hidden_size", type=int, default=256,
        help="Hidden size of LSTM of agent.")
    parser.add_argument("--batch_size", type=int, default=16,
        help="Batch size.")
    parser.add_argument("--max_trans", type=int, default=5,
        help="Max buy/sell transaction.")
    parser.add_argument("--lstm_timesteps", type=int, default=7,
        help="Number of timesteps of LSTM.")
    parser.add_argument("--n_episodes", type=int, default=100,
        help="Number of episodes.")
    parser.add_argument("--data_csv", default="data/MSFT.csv",
        help="CSV of stock data.")
    parser.add_argument("--epsilon", type=float, default=0.05,
        help="Epsilon value used for epsilon-greedy.")  
    parser.add_argument("--gamma", type=float, default=0.99,
        help="Gamma value used for future reward discount.")
    parser.add_argument("--capacity", type=int, default=100000,
        help="Maximum size of replay memory.")
    parser.add_argument("--lr", type=float, default=1e-3,
        help="Parameter learning rate.")
    parser.add_argument("--init_balance", type=float, default=10000.0,
        help="Initial balance.")
    

    main(parser.parse_args())