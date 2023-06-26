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
В качестве архитектуры для валидации был взят [LongT5](https://arxiv.org/pdf/2112.07916.pdf) с 3В параметров ([long-t5-tglobal-xl](https://huggingface.co/google/long-t5-tglobal-xl)). Брать модель большего размера было невозможно из-за ограничений по памяти. 

Я выбрала LongT5, по следующим причинам:
 - в данных присутствуеют последовательности, превосходящие 512 токенов, поэтому квадратичное внимание будет неэффективным
 - модель обучалась по стратегии PEGASUS и показала достойные результаты на суммаризации других датасетов

### 3. Гипотезы
1. Предобработанные данные (arg-filtered) дадут результат лучше vanilla.


2. Более детальный prompt с инструкциями даст результат лучше короткого. 


3. Предобученная на суммаризации книг модель даст результат на диалогах лучше базовой.


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
Модель |    Данные    | prompt | Заголовок | num_beams | Комментатор | ROUGE-1 | ROUGE-2 | ROUGE-L 
:---: |:------------:|:------:|:---------:|:---------:|:-----------:|:-------:|:-------:|:---------:
[t5-3b](https://huggingface.co/google/t5-3b) |   vanilla    |  -1-   |     +     |     1     |      -      |  19.24  |  2.86   | 11.89
[t5-3b](https://huggingface.co/google/t5-3b) |   vanilla    |  -1-   |     +     |     3     |      -      |  19.26  |  2.78   | 11.89
[t5-3b](https://huggingface.co/google/t5-3b) |   vanilla    |  -1-   |     +     |     1     |      +      |  19.20  |  2.74   | 11.92
[t5-3b](https://huggingface.co/google/t5-3b) |   vanilla    |  -2-   |     +     |     1     |      +      |  19.46  |  2.82   | 11.93
[t5-3b](https://huggingface.co/google/t5-3b) |   vanilla    |  -3-   |     +     |     1     |      +      |  19.18  |  2.76   | 11.87
[t5-3b](https://huggingface.co/google/t5-3b) |   vanilla    |  -3-   |     +     |     1     |      -      |  19.21  |  2.81   | 11.89
[t5-3b](https://huggingface.co/google/t5-3b) | arg-filtered |  -1-   |     +     |     1     |      -      |  20.34  |  3.13   | 13.85
[t5-3b](https://huggingface.co/google/t5-3b) | arg-filtered |  -1-   |     -     |     1     |      -      |  20.37  |  3.21   | 12.79
[t5-3b](https://huggingface.co/google/t5-3b) | arg-filtered |  -1-   |     -     |     3     |      -      |  20.85  |  3.20   | 13.01
[t5-3b](https://huggingface.co/google/t5-3b) | arg-filtered |  -1-   |     -     |     5     |      -      |  20.91  |  3.26   | 13.03
[book-summary](https://huggingface.co/pszemraj/long-t5-tglobal-xl-16384-book-summary) | arg-filtered |  -1-   |     -     |     3     |      -      |  25.10  |  3.98   | 15.04
[book-summary](https://huggingface.co/pszemraj/long-t5-tglobal-xl-16384-book-summary) | arg-filtered |  -1-   |     +     |     3     |      -      |  25.38  |  3.81   | 15.34
**[book-summary](https://huggingface.co/pszemraj/long-t5-tglobal-xl-16384-book-summary) |   vanilla    |  -1-   |     +     |     3     |      +      |  27.67  |  5.26   | 16.40**

-1-
```summarize: {text}``` 

-2-
```summarize the following dialogue in third person using words like commenter say and most comenneters agree: {text}```


-3-
```summarize the dialogue without using pronoun \'I\': {text}```



Результат лучшей модели на тестовой выборке:

Модель |    Данные    | prompt | Заголовок | num_beams | Комментатор | ROUGE-1 | ROUGE-2 | ROUGE-L 
:---: |:------------:|:------:|:---------:|:---------:|:-----------:|:-------:|:-------:|:---------:
**[book-summary](https://huggingface.co/pszemraj/long-t5-tglobal-xl-16384-book-summary) |   vanilla    |  -1-   |     +     |     3     |      +      |  27.02  |  4.40   | 16.05**

Так как некоторые тексты обрезались на неоконченных предложениях, дополнительно мне было интересно узнать результат для суммаризаций большей длины, а также суммаризации в две итерации (сначала в текст длины 180, затем из него в текст длины 90). Остальные параметры зафиксированы как в выделенной выше модели:


Ground Truth:
```
Commenters offer suggestions for a potential build. The Oos Crit Dagger Ranger Build was a suggestion that did not vary 
much from the original commenter's build. Incinerator is a build that several commenters agreed is good, and another 
interesting build is the Ele wander witch, which several commenters are interested in.
```


Длина 90:
```
The original poster asks for a good, not too expensive, map build. He's been playing with a dude who's level 68 
and hasn't been able to do any of the maps he's tried. He wants to try something new, but he doesn't know what to build. 
The commenters give him a bunch of good advice. The first commenter recommends
```

Длина 120:
```
The original poster asks for a good, not too expensive, map build. He's been playing with a dude who's level 68 and 
hasn't been able to do any of the maps he's tried. He wants to try something new, but he doesn't know what to build. 
The commenters give him a bunch of good advice. The first commenter recommends a crit dagger ranger build, which is 
pretty much the same as the one the original poster was using before. The second commenter
```

Все равно не хватает длины. Однако если применить две итерации, то результат становится слишком обобщенным и появляются фактические ошибки (пользователи предлагали решения со ссылками, в Гугл автора не отправляли):
```
This is a really good example of how you can take a question posted on a forum and turn it into an actual answer. 
Someone asks for map advice and the narrator gives us a long, detailed answer that basically amounts to "Google it."
```

Максимальная длина суммаризации | ROUGE-1 | ROUGE-2 | ROUGE-L 
:---:|:-------:|:-------:|:---------:
120 |  27.82  |  5.44   | 15.86
две итерации 180/90 |  24.55  |  3.53   | 14.55



#### Разбор и примеры суммаризаций
_Гипотеза 1._ 
- arg-filtered данные дали небольшой прирост по метрикам на базовой модели. Однако как с vanilla данными, где есть разделение по комментаторам, так и с arg-filtered, где его нет, модель справлялась плохо, вырезая отдельные куски исходных текстов без перевода их в третье лицо.
- на предобученной модели лучше сработали данные vanilla, так как в arg-filtered не хватало данных о комментаторе и результаты выходили от первого лица:
```
I'm new to the game, and I can't seem to find out what the maximum level you can get as a character in the game is. 
I just hit level 50 and am wondering if there's going to be a cap on how high you can go in the future.
```
vs vanilla с указанием отдельных комментаторов и оригинального постера:
```
The original poster asks what the max level is in the game right now. No one can give him an answer. 
He asks if they're planning on raising the level cap in the future. Nope. They're not.
```

_Гипотеза 2._

- более детальный промпт не показал никаких значимых изменений в работе модели; возможно, это связано с тем, что она преодобучалась на коротком ```summarize:``` и без дообучения плохо воспринимает другие промпты.


_Гипотеза 3:_

- предобученная на booksum модель показала прирост на arg-filtered данных в почти 5 пунктов ROUGE-1, 0.6 в ROUGE-2 и 2 пункта в ROUGE-L. На vanilla датасете прирост составил примерно 8.5, 2.5 и 4.5 пункта соответственно (сравнивается базовая модель без комментатора и дообученная с комментатором, но по результатам экспериментов наличие данных о комментаторе не влияло на результаты базовой модели).


Пример результата базовой модели:
```
Commenter 1: http://www.importedmods.com/shop/article_1116/Slug.html?shop_param=cid%3D3%26aid%3D1116%26 
Commenter 2: whoa whoa, Cincinnati? Commenter 4: Sexy mod btw Commenter 5: Someone is selling one on CaliVapers 
Commenter 6: I saw that
```
и лучшая по метрикам модель на том же примере:
```
The original poster of this thread is looking for a "steampunk mech" or "slug". 
He has seen a few of these advertised on various websites, but they are all sold out. He is willing to pay someone 
to help him find one. The first person to respond is a guy from Cincinnati, Ohio, who says that he just bought one 
from a place called City Vapor.
```

### 7. Выводы

На выбранную модель длина и детальность промпта влияния не оказали, для финальной модели более подходящим датасетом оказался vanilla в силу его разделенности по комментаторам, а предобученная на суммаризации книг модель дала заметный прирост в метриках по сравнению с базовой.

Повысить метрики дальше может помочь дообучение на ConvoSumm, а также добавление информации о никах комментаторов, чтобы модель получала открытую информацию о повторяющихся комментаторах в треде.