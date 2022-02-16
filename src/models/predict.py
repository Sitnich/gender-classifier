import os

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.data.loaders as ld
import src.features.feature_extraction as fe
import src.models.model_classes as cl


def model_predict(root_dir=os.path.abspath("..")):
    model, optimizer, config = ld.load_best_model(root_dir=root_dir)
    file_names, just_features = [], []
    for root, dirs, files in os.walk(root_dir + '/data/input'):
        for f in files:
            if '.wav' in f:
                feat = fe.feature_extraction((root + '/' + f), "mel")
                file_names.append(f)
                just_features.append(feat)

    train_dataset = cl.MelDataset(just_features, np.zeros(len(just_features)))
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1)

    res_list = []
    for i, (f, _) in tqdm(enumerate(train_dataloader)):
        f = f.reshape((1, 1, 128))
        pred = model(f).item()
        sex = 'female' if pred >= 0.5 else 'male'
        prob = max(pred, 1 - pred)

        res_list.append((file_names[i], sex, prob * 100))

    return res_list
