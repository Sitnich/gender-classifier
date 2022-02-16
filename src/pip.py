import os

import src.data.loaders as ld
import src.data.prepare_data as pp
import src.features.feature_extraction as fe
import src.models.predict as prd
from src.data import speaker_info_extractor as clx

root_dir = os.path.abspath("..")


def process():
    clx.extract_classes()

    if not os.path.isdir(root_dir + '/data/train100/'):
        pp.unzip_train()
    if not os.path.isdir(root_dir + '/data/test/'):
        pp.unzip_test()

    dict_train_path, dict_test_path, train_classes, test_classes = pp.extract_paths()
    fe.extract_features(dict_train_path, train_classes, 'train')
    fe.extract_features(dict_test_path, test_classes, 'test')


def print_model_info():
    model, optimizer, config = ld.load_best_model()
    print('---------------------- Model info ---------------------- \n {} '.format(model))
    print('-------------------- Optimizer info -------------------- \n {}'.format(optimizer))
    print('-------------- Learning configuration info ------------- \n {}'.format(config))


if __name__ == '__main__':
    prd.model_predict()
