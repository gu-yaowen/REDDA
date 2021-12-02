import os
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-id', '--device_id', default=None, type=int,
                    help='Set the device (GPU ids).')
parser.add_argument('-data', '--dataset', type=str,
                    help='Set the data set for training.')
parser.add_argument('-path', '--saved_path', type=str,
                    help='Path to save training results')
parser.add_argument('-seed', '--seed', default=0, type=int,
                    help='Global random seed')
parser.add_argument('-epoch', '--epoch', default=100, type=int,
                    help='Number of epochs for training')
parser.add_argument('-batch', '--batch_size', default=32, type=int,
                    help='batch size to use')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='learning rate to use')
parser.add_argument('-wd', '--weight_decay', default=0.0, type=float,
                    help='weight decay to use')

args = parser.parse_args()
args.saved_path = args.saved_path + '_' + str(args.seed)