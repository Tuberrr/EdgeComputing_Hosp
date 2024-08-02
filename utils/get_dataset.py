from torchvision import datasets, transforms
import numpy as np
import random
from collections import Counter

def separate_data(train_data, num_clients, num_classes, beta=0.4):

    y_train = np.array(train_data.targets)

    min_size_train = 0
    min_require_size = 10
    K = num_classes

    N_train = len(y_train)
    dict_users_train = {}

    while min_size_train < min_require_size:
        idx_batch_train = [[] for _ in range(num_clients)]
        idx_batch_test = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k_train = np.where(y_train == k)[0]
            np.random.shuffle(idx_k_train)
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            proportions_train = np.array([p * (len(idx_j) < N_train / num_clients) for p, idx_j in zip(proportions, idx_batch_train)])
            proportions_train = proportions_train / proportions_train.sum()
            proportions_train = (np.cumsum(proportions_train) * len(idx_k_train)).astype(int)[:-1]
            idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(idx_k_train, proportions_train))]
            min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
            # if K == 2 and n_parties <= 10:
            #     if np.min(proportions) < 200:
            #         min_size = 0
            #         break

    for j in range(num_clients):
        np.random.shuffle(idx_batch_train[j])
        dict_users_train[j] = idx_batch_train[j]

    return dict_users_train

def get_public(dataset, ratio):
    data_volume = len(dataset)
    all_idxs = [i for i in range(len(dataset))]
    dict_public = set(np.random.choice(all_idxs, data_volume*ratio, replace=False))
    dict_public = np.array(list(dict_public)).tolist()
    
    return dict_public


def get_hosp(args, dataset, hosp_datavolume):
    all_idxs = [i for i in range(len(dataset))]
    sum_hosp_datavolume = sum(hosp_datavolume)
    dict_users = separate_data(
                    dataset, sum_hosp_datavolume, args.num_classes, args.data_beta
                )
    dict_hosp = [[] for _ in len(hosp_datavolume)]
    for idx_hosp in range(len(hosp_datavolume)):
        for idx in range(hosp_datavolume[idx_hosp]):
            dict_hosp[idx_hosp] += dict_users[idx]
        dict_users = dict_users[hosp_datavolume[idx_hosp]:]
    return dict_hosp
    
def get_dataset(hosp_datavolume):

    trans_cifar10_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trans_cifar10_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset_train = datasets.CIFAR10(
        "./data/cifar10", train=True, download=True, transform=trans_cifar10_train
    )
    dataset_test = datasets.CIFAR10(
        "./data/cifar10", train=False, download=True, transform=trans_cifar10_val
    )
    dict_public = get_public(dataset_train, ratio = 0.6)
    dict_hosp = get_hosp(dataset_train, hosp_datavolume)
    return dataset_train, dataset_test, dict_public, dict_hosp
