import os
import pickle

import numpy as np
import torch
from sklearn.utils import shuffle
from torch import optim

import src.models.model_classes as cl


# методы загрузки признаков X и меток Y для всех аудиофайлов
# из обучающей и тестовой выборок соответственно

def load_data_train(root_dir=os.path.abspath("..")):
    X = np.load(root_dir + "/data/all_train_audio_features.npy")
    Y = np.load(root_dir + "/data/all_train_labels.npy")

    return X, Y


def load_data_test(root_dir=os.path.abspath("..")):
    X = np.load(root_dir + "/data/all_test_audio_features.npy")
    Y = np.load(root_dir + "/data/all_test_labels.npy")

    return X, Y


# метод загрузки получившейся предобученной модели и оптимизатора
def load_best_model(root_dir=os.path.abspath(".."), inp_size=128):
    best_trained_model = cl.Model(inp_size)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    best_trained_model.to(device)

    with open(root_dir + "/models/config.txt", "rb") as fp:
        config = pickle.load(fp)

    best_trained_model.load_state_dict(torch.load(root_dir + '/models/cnn_model.npy'))

    optimizer = optim.SGD(best_trained_model.parameters(), lr=config['lr'])
    optimizer.load_state_dict(torch.load(root_dir + '/models/optim_params.npy'))

    return best_trained_model, optimizer, config


def load_train_test():
    X_train, Y_train = load_data_train()
    X_test, Y_test = load_data_test()

    np.random.seed(42)
    X_train, Y_train = shuffle(X_train, Y_train)
    X_test, Y_test = shuffle(X_test, Y_test)
    return X_train, Y_train, X_test, Y_test
