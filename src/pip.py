import os
import src.data.prepare_data as pp
import src.features.feature_extraction as fe
import src.data.loaders as ld
from src.data import speaker_info_extractor as clex
root_dir = os.path.abspath("..")


def process():
    clex.extract_classes()

    if not os.path.isdir(root_dir + '/data/train100/'):
        pp.unzip('train')
    if not os.path.isdir(root_dir + '/data/test/'):
        pp.unzip('test')

    dict_train_path, dict_test_path, train_classes, test_classes = pp.extract_paths()
    fe.extract_features(dict_train_path, train_classes, 'train')
    fe.extract_features(dict_test_path, test_classes, 'test')
def model_predict():
   model, optimizer = ld.load_best_model()

def print_model_info():
   model, optimizer = ld.load_best_model()
   print('---------------------- Model info ---------------------- \n {} '.format(model))
   print('---------------------- Optimizer info ---------------------- \n {}'.format(optimizer))


if __name__ == '__main__':
    print_model_info()
