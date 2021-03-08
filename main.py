import argparse
from trainer import Trainer
def main(args):
    model_trainer = Trainer(args)
    model_trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true",
        help="Load saved models.")
    parser.add_argument("--hidden_size", type=int, default=256,
        help="Hidden size of LSTM of agent.")
    parser.add_argument("--max_trans", type=int, default=5,
        help="Max buy/sell transaction.")
    parser.add_argument("--lstm_timesteps", type=int, default=7,
        help="Number of timesteps of LSTM.")
    parser.add_argument("--data_csv", default="data/MSFT.csv",
        help="CSV of stock data.")
    
    

    main(parser.parse_args())