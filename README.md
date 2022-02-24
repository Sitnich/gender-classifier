## Классификатор мужских и женских голосов для проекта Huawei "Детектирование и синтез эмоций для задач голосового ассистента"

Полный процесс подготовки данных, извлечения признаков, построения модели,
её обучения и настройки гиперпараметров представлен в ноутбуке voice_classifier.ipynb
в директории `\notebooks`. Отчет об анализе данных и экспериментах лежит в `\reports`
в файле analysis_experiments.pdf.

Для дальнейшей работы нужно:
1) Активировать виртуальную среду: `.\venv\Scripts\activate`
2) Установить зависимости: `pip install -r requirements.txt`

Далее для взаимодействия с моделью можно воспользоваться файлом cli.py: `python3 workflow\scripts\cli.py`
```
    Usage: cli.py [OPTIONS] COMMAND [ARGS]...
    
    Options:
      --help  Show this message and exit.
    
    Commands:
      model-info  выводит информацию о структуре CNN модели, оптимизатора и...
      predict     предсказывает пол спикера по записи в формате .wav
      process     распаковывает zip-файлы и извлекает признаки
      train       обучает модель по заданной конфигурации

```
Чтобы проверить работу модели нужно положить wav-файлы для предсказания в директорию 
\data\input\  и запустить команду **predict**: ``` python3 workflow\scripts\cli.py predict --help```

Чтобы распаковать и извлечь признаки из обучающей и тестовой выборок LibriTTS, нужно сперва их
загрузить из сторонних источников (github не позволяет загружать объемные файлы), а затем
воспользваться командой **process**: ``` python3 workflow\scripts\cli.py process --help ```

Для вывода информации о модели и оптимизаторе поможет команда
**model-info**.

Чтобы обучить модель с заданной конфигурацией (по дефолту стоит \models\config.txt)
можно воспользоваться командой **train**.
