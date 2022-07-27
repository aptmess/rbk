# Цифровой прорыв 2022. [Задача «Радар тенденций новостных статей»](https://lk.hacks-ai.ru/758453/champ/768812)

## Author

Александр Широков

- `t.me/aptmess`

## Задача

Цель модели участников — предсказать 3 численные характеристики, которые в полной мере показывают популярность статьи: 

- `views` - количество просмотров
- `full_reads_percent` - процент читателей полностью прочитавших статью
- `depth` - объем прочитанного материала

Для оценки качества решения используется метрика `R2`:

`result = 0.4 * R2_views + 0.3 * R2_full_reads_percent + 0.3 * R2_depth`

## Решение

Итоговое решение состоит из следующих этапов:

1. Парсинг дополнительных данных с сайта РБК на страницах новостей, используя `document_id`:
    - `make parse_news_train` - парсинг для `train` датасета: `./data/parsed/train_parsed.json`
    - `mask parse_news_test` - парсинг для `test` датасета: `./data/parsed/test_parsed.json`
Были собраны дополнительные признаки такие как текст новости, имена и информация об авторах и тэгах, информация о длине текста, наличии изображения в статье итд.
Данные о просмотрах или какие-то статистические данные не были использованы. Все данные парсятся автоматически с помощью скрипта выше.

2. `Data Preprocessing` - используется модуль `src.transformers.preprocess` для препроцессинга данных

   - `./data/full/full_train.json` - тренировочные данные после процессинга
   - `./data/full/full_test.json` - тестовые данные после процессинга

```python
import pandas as pd

from src.transformers.base import Compose
from src.transformers.preprocess import (
    LoaderMergePreprocess, 
    CategoryFromTextPreprocess, 
    AuthorsPreprocess,
    TagsPreprocess,
    FeaturePreprocess,
    SaverPreprocess,
    NatashaTransformer
)

Preprocessor = Compose(
    transforms=[
        LoaderMergePreprocess(name='loading'),
        CategoryFromTextPreprocess(name='category from text'),
        AuthorsPreprocess(name='authors preprocess'),
        TagsPreprocess(name='tags preprocess'),
        FeaturePreprocess(name='feature selector'),
        NatashaTransformer(name='natasha name entity'),
        SaverPreprocess(name='saving files')
    ]
)

train = Preprocessor(data=pd.DataFrame(), mode='train')
test = Preprocessor(data=pd.DataFrame(), mode='test')
```

3. `Feature Generation` - генерация признаков. Используются `TFIDF`-признаки для `title, authors и tags`, признаки из времени и собранные признаки при парсинге.
4. `Fit Models` - валидация по трем фолдам, преобразование таргета путем логирования, использование `LightGBM` регрессора и усреднение значения для предсказания
   - `views: 70.8+-0.3`
   - `depth: 83.9+-0.3`
   - `full_reads_percent: 55.1+-0.4`

5. `Sub` - предсказание + отдельно предсказания для данных, которые есть и в `train` и в `test`

## Запуск проекта локально

### Make Venv

1. Создать виртуальное окружение в `PyCharm` в папке `.venv`
2. Запустить команду `make venv`
3. Установка `Jupyter kernel`: `python -m ipykernel install --name rbk --user`

### Получение предсказания

Необходимо запустить notebook `./notebook/full_pipeline.ipynb`. 

Лучшее решение получилось здесь: `./notebooks/step-2.ipynb`

## Результат

`public result = 0.7619`