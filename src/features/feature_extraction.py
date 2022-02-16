import os
import warnings

import librosa
import librosa.display
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

root_dir = os.path.abspath("..")


def feature_extraction(file_name, feature_name):
    # метод для извлечения признаков из аудиозаписи
    # принимает аргументы file_name: путь до аудиофайла,
    # и feature_name: название набора признаков, которые можно извлечь
    # (допустимые значения: ["mfcc", "mel", "chroma", "tonnetz", "contrast"])

    audio, sample_rate = librosa.core.load(file_name)

    if feature_name == "mfcc":
        features = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=25).T, axis=0)
    if feature_name == "chroma":
        EnrgSpectra = np.abs(librosa.stft(audio))
        features = np.mean(librosa.feature.chroma_stft(S=EnrgSpectra, sr=sample_rate).T, axis=0)
    if feature_name == "mel":
        features = np.mean(librosa.feature.melspectrogram(audio, sr=sample_rate).T, axis=0)
    if feature_name == "tonnetz":
        features = np.mean(librosa.feature.tonnetz(y=audio, sr=sample_rate).T, axis=0)
    if feature_name == "contrast":
        features = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)

    features = preprocessing.scale(features)

    return features


def extract_features(dict_train_path, train_classes, file):
    feature_list, label, features = [], [], []

    # извлекаем признаки для все аудиозаписей каждого из спикеров
    # и кладем признаки в лист features, а соответствующие метки - в список label

    for speaker, sex in tqdm(train_classes):
        for f in dict_train_path[speaker]:
            features = feature_extraction(f, "mel")
            feature_list.append(features)
            if sex == 0:
                label.append(1)
            else:
                label.append(0)

    Y = np.array(label).reshape(len(label), 1)
    X = np.array(feature_list).reshape(len(feature_list), len(features))

    if file == 'train':
        np.save(root_dir + "/data/all_train_audio_features.npy", X)
        np.save(root_dir + "/data/all_train_labels.npy", Y)
    if file == 'test':
        np.save(root_dir + "/data/all_test_audio_features.npy", X)
        np.save(root_dir + "/data/all_test_labels.npy", Y)


