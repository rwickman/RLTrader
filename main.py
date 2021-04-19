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
    parser.add_argument("--batch_size", type=int, default=32,
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
    parser.add_argument("--lr", type=float, default=1e-4,
        help="Parameter learning rate.")
    parser.add_argument("--max_init_balance", type=float, default=10000.0,
        help="Initial balance (also max if randomizing balance).")
    parser.add_argument("--min_init_balance", type=float, default=100.0,
        help="Minimum initial balance.")
    parser.add_argument("--max_train_days", type=int, default=256,
        help="The maximum amount of days to train the agent.")
    parser.add_argument("--min_train_days", type=int, default=8,
        help="The minimum amount of days to train the agent.")
    parser.add_argument("--no_rand_balance", action="store_true",
        help="Don't randomize the balance.")
    parser.add_argument("--no_rand_start", action="store_true",
        help="Don't randomize start training day.")
    parser.add_argument("--no_rand_days", action="store_true",
        help="Don't randomize the number of training day.")
    parser.add_argument("--tgt_update", type=int, default=1,
        help="Number of episode elapsed before target network is updated")

    parser.add_argument("--eps", type=float, default=1e-6,
        help="Epsilon used for proportional priority.")
    parser.add_argument("--alpha", type=float, default=0.6,
        help="Alpha used for proportional priority.")
    parser.add_argument("--beta", type=float, default=0.4,
        help="Beta used for proportional priority.")
    parser.add_argument("--grad_clip", type=float, default=1.0,
        help="Gradient clipping.")


    main(parser.parse_args())