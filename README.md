# Суммаризация диалогов

### 0. requirements.txt
Все необходимые библиотеки и их версии лежат в [requirements.txt]():

```
pip install -r requirements.txt
```

### 1. Данные
Использовались данные Reddit из датасета [ConvoSumm](https://github.com/Yale-LILY/ConvoSumm).

Для возможности сравнения последующего дообучения (*спойлер*: на него в итоге не хватило ресурсов, поэтому далее дообучение рассматриваться не будет) с результатами авторов использовалось то же разделение ```train/val/test```, что и в [публикации](https://arxiv.org/pdf/2106.00829.pdf). Однако в полном датасете (```reddit.all.newsumms.*```) встречались лишние newline символы, из-за которых в оригинальном разделении были примеры, состоящие из одного сообщения, оторванного от оригинального треда. В самих суммаризациях (файлы ```*.target```) такой проблемы не было обнаружено, поэтому я сделала свое разделение на выборки, опираясь на индексы примеров в ```train.target``` и ```val.target```. Код находится в [preprocess.py](), скрипт для запуска:

```
cd data
python preprocess.py --path_to_data <path to reddit vanilla>
```

### 2. Модель_и
В качестве первой модели для валидации был взят [LongT5](https://arxiv.org/pdf/2112.07916.pdf) с 3В параметров ([long-t5-tglobal-xl](https://huggingface.co/google/long-t5-tglobal-xl)). Брать модель большего размера было невозможно из-за ограничений по памяти. 

Я выбрала LongT5, по следующим причинам:
 - в данных присутствуеют последовательности, превосходящие 512 токенов, поэтому квадратичное внимание будет неэффективным
 - модель обучалась по стратегии PEGASUS и показала достойные результаты на суммаризации других датасетов

### 3. Гипотезы
1. Предобработанные данные (arg-filtered) дадут результат лучше vanilla.


2. Наличие заголовков во входных данных улучшит результат.



3. Более детальный prompt с инструкциями даст результат лучше короткого. 


4. Предобученная на суммаризации книг модель даст результат на диалогах лучше базовой.


### 4. Метрики
В качестве колличественной метрики был взят ROUGE1/2/L, так как, во-первых, ..., и во-вторых, она использовалась в статье ConvoSumm, что позволит сравнить результаты.


### 5. Эксперименты
Все регулируемые в экспериментах гиперпараметры находятся в файлах [data_params.json](https://github.com/dsashulya/summarization_test_task/blob/main/src/params/data_params.json), [validation_params.json](https://github.com/dsashulya/summarization_test_task/blob/main/src/params/validation_params.json) и [prompt.txt](https://github.com/dsashulya/summarization_test_task/blob/main/src/params/prompt.txt).
Все возможные комбинации изучить не было возможности, поэтому я начала с базовой комплектации модели (без дообучения, жадный декодинг без beam search), и после изменения отдельных параметров продолжала работать с моделью, показавшей себя лучше.

Эксперименты проводились на NVIDIA A100-SXM4-40GB в Google Colab.


Скрипт для запуска лучшей модели (*в файлах с параметрами необходимо указать директории с данными и для сохранения результатов модели*):
```
cd src
python validate.py
```
Изначально деление на ```train\val``` было оставлено для потенциального дообучения, но в итоге не пригодилось. В скрипте метрики считаются отдельно для этих датасетов, но в таблице ниже они объединены.

### 6. Результаты

#### Метрики
Модель |    Данные    |                                                          prompt                                                          | Заголовок | num_beams | Комментатор | ROUGE-1 | ROUGE-2 | ROUGE-L 
:---: |:------------:|:------------------------------------------------------------------------------------------------------------------------:|:---------:|:---------:|:-----------:|:-------:|:-------:|:---------:
[google/t5-3b](https://huggingface.co/google/t5-3b) |   vanilla    |                                                 ```summarize: {text}```                                                  |     +     |     1     |      -      |  19.24  |  2.86   | 11.89
[google/t5-3b](https://huggingface.co/google/t5-3b) |   vanilla    |                                                 ```summarize: {text}```                                                  |     +     |     3     |      -      |  19.26  |  2.78   | 11.89
[google/t5-3b](https://huggingface.co/google/t5-3b) |   vanilla    |                                                 ```summarize: {text}```                                                  |     +     |     1     |      +      |  19.20  |  2.74   | 11.92
[google/t5-3b](https://huggingface.co/google/t5-3b) |   vanilla    | ```summarize the following dialogue in third person using words like commenter say and most comenneters agree: {text}``` |     +     |     1     |      +      |  19.46  |  2.82   | 11.93
[google/t5-3b](https://huggingface.co/google/t5-3b) |   vanilla    |                             ```summarize the dialogue without using pronoun \'I\': {text}```                             |     +     |     1     |      +      |  19.18  |  2.76   | 11.87
[google/t5-3b](https://huggingface.co/google/t5-3b) |   vanilla    |                             ```summarize the dialogue without using pronoun \'I\': {text}```                             |     +     |     1     |      -      |  19.21  |  2.81   | 11.89
[google/t5-3b](https://huggingface.co/google/t5-3b) | arg-filtered |                                                 ```summarize: {text}```                                                  |     +     |     1     |      -      |  20.34  |  3.13   | 13.85
[google/t5-3b](https://huggingface.co/google/t5-3b) | arg-filtered |                                                 ```summarize: {text}```                                                  |     -     |     1     |      -      |  20.37  |  3.21   | 12.79
[google/t5-3b](https://huggingface.co/google/t5-3b) | arg-filtered |                                                 ```summarize: {text}```                                                  |     -     |     3     |      -      |  20.85  |  3.20   | 13.01
[google/t5-3b](https://huggingface.co/google/t5-3b) | arg-filtered |                                                 ```summarize: {text}```                                                  |     -     |     5     |      -      |  20.91  |  3.26   | 13.03
[pszemraj/long-t5-tglobal-xl-16384-book-summary](https://huggingface.co/pszemraj/long-t5-tglobal-xl-16384-book-summary) | arg-filtered |                                                 ```summarize: {text}```                                                  |     -     |     3     |      -      |  25.10  |  3.98   | 15.04
[pszemraj/long-t5-tglobal-xl-16384-book-summary](https://huggingface.co/pszemraj/long-t5-tglobal-xl-16384-book-summary) | arg-filtered |                                                 ```summarize: {text}```                                                  |     +     |     3     |      -      |  25.38  |  3.81   | 15.34
[pszemraj/long-t5-tglobal-xl-16384-book-summary](https://huggingface.co/pszemraj/long-t5-tglobal-xl-16384-book-summary) |   vanilla    |                                                 ```summarize: {text}```                                                  |     +     |     3     |      +      |  27.67  |  5.26   | 16.40
[pszemraj/long-t5-tglobal-xl-16384-book-summary](https://huggingface.co/pszemraj/long-t5-tglobal-xl-16384-book-summary) |   vanilla    |                                                 ```summarize: {text}```                                                  |     +     |     3     |    + 120    |  27.82  |  5.44   | 15.86
[pszemraj/long-t5-tglobal-xl-16384-book-summary](https://huggingface.co/pszemraj/long-t5-tglobal-xl-16384-book-summary) |   vanilla    |                                                 ```summarize: {text}```                                                  |     +     |     3     |  + 180/90   |  24.55  |  3.53   | 14.55

#### Примеры суммаризаций
Для booksum модели в ```arg-filtered`` не хватало данных о комментаторе.