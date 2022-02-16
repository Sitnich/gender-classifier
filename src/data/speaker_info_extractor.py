import os
import re

import numpy as np


def extract_classes():
    root_dir = os.path.abspath("..")

    list_train, list_test = np.empty((0, 2)), np.empty((0, 2))
    infile = open(root_dir + '\data\libritts_speakerinfo.txt')
    lines = infile.readlines()
    for i, line in enumerate(lines):
        if 'train-clean-100' in line:
            line_list = re.split("\n|\|", line)[:2]
            stripped = [s.strip() for s in line_list]
            if stripped[1] == 'F':
                stripped[1] = 0
            else:
                stripped[1] = 1

            list_train = np.append(list_train, np.array([stripped]), axis=0)
        if 'test-clean' in line:
            line_list = re.split("\n|\|", line)[:2]
            stripped = [s.strip() for s in line_list]
            if stripped[1] == 'F':
                stripped[1] = 0
            else:
                stripped[1] = 1

            list_test = np.append(list_test, np.array([stripped]), axis=0)

    with open(root_dir + '\data\\train_speaker_info.npy', 'wb') as f_train:
        np.save(f_train, list_train)
    with open(root_dir + '\data\\test_speaker_info.npy', 'wb') as f_test:
        np.save(f_test, list_test)
