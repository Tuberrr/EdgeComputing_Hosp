#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=5, help="rounds of training")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--num_classes', type=int, default=10, help="number of channels of imges")
    parser.add_argument('--data_beta', type=float, default=1.0,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--num_users', type=int, default=161, help="number of users: K")

    args = parser.parse_args()
    return args
