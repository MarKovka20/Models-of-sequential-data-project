# Research of efficient transformers

**Автор:** Ковалева Мария

## Постановка задачи

Методы, основанные на внимании и трансформеры [3] совершили значительный прорыв в области глубокого обучения и оказали значительное влияние на решение задач НЛП.
Применяя механизмы внимания, можно столкнуться с очевидным ограничением длины входной последовательности из-за квадратичной сложности метода. В недавних статьях предлагались различные способы преодоления этой проблемы, но мы хотим сосредоточиться на двух перспективных подходах: informer и performer [1, 2].
Основное предположение  Informer-а заключается в том, что модель должна обладать "бесконечной памятью" и соответствовать последовательности произвольной длины. Модель performer показывает хорошие результаты в задаче NLP, но недостаточно изучена для других типов данных. Его основная идея состоит в том, чтобы использовать некоторую тригонометрическую аппроксимацию матрицы внимания для уменьшения потребления памяти.
Таким образом Целью работы являтеся сравнение нескольких недавних методов, предложенных для снижения сложности оценки в конкретных задачах прогнозирования пола пользователя на основе транзакций. 

[1] Martins, Pedro Henrique, Zita Marinho, and André FT Martins. "oo-former: Infinite Memory Transformer." - "Informer" model
[2] Choromanski, Krzysztof, et al. "Rethinking attention with performers." arXiv preprint arXiv:2009.14794 (2020). - "Performer" model
[3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. - "Full attention" model

## Исходный код

Исходники кода находятся в [./src](./code) .  [main.py](./code/main.py) содержит запуск экспериментов.
[experiment.py](./code/experiment.py) содержит реализацию шаблона проведения эксперимента.
Данные сохраняются при помощи [results.py](./code/results.py) для каждого проведённого эксперимента.
[mathmodel.py](./code/mathmodel.py) cодержит основные компоненты для провдения экспериментов. .

## Что сделано
 - [ ] реализована baseline модель трансформера для данных транзакций 
 - [ ] реализована модель performer для данных транзакций
 - [ ] реализован простейшая модель informer для данных транзакций
 - [ ] реализован пайплайн обучения

## Что осталось сделать

 - [ ] отладить модели, подобрать гиперпараметры
 - [ ] запустить эксперименты для всех моделей
 - [ ] сравнить результаты, скорость и потребление памяти всех моделей
