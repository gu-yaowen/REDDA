import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# General Arguments
parser.add_argument('-id', '--device_id', default=None, type=str,
                    help='Set the device (GPU ids).')
parser.add_argument('-da', '--dataset', type=str, choices=['Bdataset', 'Kdataset'],
                    help='Set the data set for training.')
parser.add_argument('-sp', '--saved_path', type=str,
                    help='Path to save training results', default='result')
parser.add_argument('-se', '--seed', default=42, type=int,
                    help='Global random seed')
# Training Arguments
parser.add_argument('-fo', '--nfold', default=10, type=int,
                    help='The number of k in K-folds Validation')
parser.add_argument('-ep', '--epoch', default=1000, type=int,
                    help='Number of epochs for training')
parser.add_argument('-lr', '--learning_rate', default=0.005, type=float,
                    help='learning rate to use')
parser.add_argument('-wd', '--weight_decay', default=0.0, type=float,
                    help='weight decay to use')
parser.add_argument('-pa', '--patience', default=100, type=int,
                    help='Early Stopping argument')
# Model Arguments
parser.add_argument('-hf', '--hidden_feats', default=64, type=int,
                    help='The dimension of hidden tensor in the model')
parser.add_argument('-he', '--num_heads', default=5, type=int,
                    help='Number of attention heads the model has')
parser.add_argument('-dp', '--dropout', default=0.0, type=float,
                    help='The rate of dropout layer')

args = parser.parse_args()
args.saved_path = args.saved_path + '_' + str(args.seed)
