
import argparse
from lstm import text_LSTM


#%%



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This A.I. Rapper Needs Training')
    parser.add_argument('artist_name', metavar='artist', type=str,
                        help='Artist to be mimicked. If you are hesitating, type "박재범" or "염따".')
    parser.add_argument('--epochs', type=int, default=3, required=False,
                        help='Train epochs')       
    args = parser.parse_args()
    

    t1 = text_LSTM(args.artist_name)
    t1.func_make_structure()
    t1.func_train(args.epochs)
    