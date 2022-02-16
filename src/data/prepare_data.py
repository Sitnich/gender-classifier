import os
import zipfile

import numpy as np
from tqdm import tqdm

root_dir = os.path.abspath("..")


def unzip(file):
    if file == 'train':
        with zipfile.ZipFile(root_dir + '/data/train-clean-100.zip', 'r') as zip_ref:
            zip_ref.extractall(root_dir + '/data/train100/')
    if file == 'test':
        with zipfile.ZipFile(root_dir + '/data/test-clean.zip', 'r') as zip_ref:
            zip_ref.extractall(root_dir + '/data/test/')


def extract_paths():
    with open(root_dir + '/data/train_speaker_info.npy', 'rb') as f_train:
        train_classes = np.load(f_train)
    with open(root_dir + '/data/test_speaker_info.npy', 'rb') as f_test:
        test_classes = np.load(f_test)
    train_classes = train_classes.astype(int)
    train_speakers = train_classes[:, 0]

    test_classes = test_classes.astype(int)
    test_speakers = test_classes[:, 0]

    # сохраним в словарь dict_train_path все пути к аудиофайлам обучающей выборки
    # (для каждого ключа-id спикера храним список путей до их аудиозаписей)

    dict_train_path = {}
    for speaker in tqdm(train_speakers):
        dict_train_path[speaker] = []
        for root, dirs, files in os.walk(root_dir + '/data/train100/' + str(speaker)):
            for f in files:
                if '.wav' in f:
                    dict_train_path[speaker].append(root + '/' + f)

    # то же самое для тестовой выборки

    dict_test_path = {}
    for speaker in tqdm(test_speakers):
        dict_test_path[speaker] = []
        for root, dirs, files in os.walk(root_dir + '/data/test/test-clean/' + str(speaker)):
            for f in files:
                if '.wav' in f:
                    dict_test_path[speaker].append(root + '/' + f)

    return dict_train_path, dict_test_path, train_classes, test_classes

