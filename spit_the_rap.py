
import argparse
from lstm import text_LSTM


#%%



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This A.I. Rapper Needs Training')
    parser.add_argument('artist_name', metavar='artist', type=str,
                        help='Artist to be mimicked. You can type "박재범" or "염따".')
    parser.add_argument('--start_words', type=str, default="난 배가 고파", required=False,
                        help='Say start words, then this rapper will rap something.')   
    parser.add_argument('--temperature', type=float, default=3, required=False,
                        help='When temperature get higher, this rapper says random words more.')   
    parser.add_argument('--epochs', type=int, default=300, required=False,
                        help='Load models of specific epochs.')       
    parser.add_argument('--show_prob', type=bool, default=False, required=False,
                        help='Show probability of next words.')       
    args = parser.parse_args()    

    t1 = text_LSTM(args.artist_name)
    t1.func_make_structure()
    
    t1.func_load(epochs_load = args.epochs)
    t1.func_rap(args.start_words, args.temperature)
    
    if args.show_prob:
        t1.func_show_prob(args.start_words)
    