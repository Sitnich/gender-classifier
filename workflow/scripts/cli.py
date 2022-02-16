import os
import sys

root_dir = os.path.abspath("")
sys.path.append(root_dir)

if os.path.basename(os.path.normpath(root_dir)) == 'scripts':
    root_dir = os.path.abspath("..\..")
    sys.path.append(root_dir)

import src.models.predict as prd
import src.data.loaders as ld
import src.features.feature_extraction as fe
import src.data.prepare_data as pp
from src.data import speaker_info_extractor as clx

import click


@click.group()
def cli():
    pass


@cli.command()
@click.option("--paths", "-p", type=click.Path(), nargs=2, help='пути до zip-файлов с обучающей и тестовой выборками')
def process(paths):
    """
    распаковывает zip-файлы с обучающей и тестовой выборками \n
    в директории /data/train100/ и /data/test/ \n
    и извлекает mel-признаки и метки для них \n
    (сохраняет их в /data/all_train_audio_features.npy и в /data/all_train_labels.npy)
    """
    clx.extract_classes()

    if not os.path.isdir(root_dir + '/data/train100/'):
        pp.unzip_train(paths[0])
    if not os.path.isdir(root_dir + '/data/test/'):
        pp.unzip_test(paths[1])

    dict_train_path, dict_test_path, train_classes, test_classes = pp.extract_paths()
    fe.extract_features(dict_train_path, train_classes, 'train')
    fe.extract_features(dict_test_path, test_classes, 'test')


@cli.command()
@click.option('--path', '-p', default=root_dir + '\\reports\\model_info.txt', type=click.Path(),
              help='путь для сохранения вывода')
def model_info(path):
    """
    выводит информацию о структуре CNN модели, оптимизатора и гиперпараметров
    """
    model, optimizer, config = ld.load_best_model(root_dir=root_dir)
    info = '---------------------- Model info ---------------------- \n {} '.format(model) + \
           '-------------------- Optimizer info -------------------- \n {}'.format(optimizer) + \
           '-------------- Learning configuration info ------------- \n {}'.format(config)
    with open(path, 'w') as f:
        f.write(info)
    print(info)


@cli.command()
@click.option("--path", '-p', default=root_dir + '\\reports\\prediction_info.txt', type=click.Path(),
              help='путь для сохранения вывода')
def predict(path):
    """
    предсказывает пол спикера по записи в формате .wav \n
    добавьте файлы для предсказания в директорию /data/input/
    """

    res = prd.model_predict(root_dir=root_dir)
    info = ''
    for items in res:
        info += 'file "{}": {} with confidence {:.4f} % \n'.format(*items)

    with open(path, 'w') as f:
        f.write(info)
    print(info)


if __name__ == "__main__":
    cli()
