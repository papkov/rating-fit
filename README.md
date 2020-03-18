# rating-fit
(English below)

[rating.chgk.info](rating.chgk.info) &mdash; сайт рейтинга 
[спортивной версии игры "Что? Где? Когда?"](https://ru.wikipedia.org/wiki/Что%3F_Где%3F_Когда%3F_(спортивная_версия)). 
Рейтинг команд там рассчитывается по [эмпирической формуле](http://mak-chgk.ru/komissii/rating/polozhenie2020/) и имеет 
очевидную проблему: он ориентирован на команды с фиксированным составом и плохо справляется с предсказанием мест сборных.

Описанный ниже метод рассматривает команду как комбинацию игроков, которых расставляет по уровню игры.


## Использование

```
python main.py --help
usage: rating [-h] [--name NAME] [--checkpoint-path CHECKPOINT_PATH]
              [--save-each] [--date-start DATE_START] [--date-end DATE_END]
              [--tournaments TOURNAMENTS] [--lr LR] [--wd WD]
              [--momentum MOMENTUM] [--loss {logsigmoid,sigmoid}]
              [--clip-zero]

Model training for player ratings

optional arguments:
  -h, --help            show this help message and exit
  --name NAME, -n NAME  Model name to save checkpoint
  --checkpoint-path CHECKPOINT_PATH
                        Dir to save checkpoints to
  --save-each           Save each epoch (tournament) checkpoint in separate
                        file
  --date-start DATE_START, -s DATE_START
                        Use tournaments starting this date
  --date-end DATE_END, -e DATE_END
                        Use tournaments until this date
  --tournaments TOURNAMENTS
                        Use preloaded list of tournaments
  --lr LR               Learning rate
  --wd WD               Weight decay
  --momentum MOMENTUM   Momentum
  --loss {logsigmoid,sigmoid}
                        Loss function to train a model
  --clip-zero           Shift model embeddings so that min == 0

```

## Алгоритм
Каждый турнир представлен в виде набора микроматчей: в турнире принимает участие _N_ команд, каждая из них играет 
_N-1_ микроматчей со всеми остальными командами с возможными результатами {-1, 0, 1} (поражение, ничья, победа). 
Количество взятых вопросов и разница мест никак не учитывается, потому что сложность вопросов и массовость турниров 
разнятся. 

Каждый игрок представлен эмбеддингом (по сути это рейтинг игрока). Эмбеддинг можно получить по индексу (id) игрока.

Для каждого микроматча в турнире рассчитывается функция потерь и ее градиент, чтобы обновить эмбеддинги (рейтинги) игроков:
1. Для всех игроков каждой команды достаются их эмбеддинги, рассчитывается их среднее значение. Это значение &mdash;
рейтинг команды (_S_). 
2. В зависимости от результата микроматчи оптимизируется разница рейтингов команд (_D=S1-S2_). Если _S1 > S2_ ожидается 
результат _R = 1_, если _S2 > S1_, ожидается результат _R = -1_. Разница _D_ идет в сигмоидную функцию (или ее логарифм, варианты ниже), 
в результате мы получаем функцию потерь.
3. Для эмбеддингов игроков рассчитывается градиент по функции потерь


<img src="https://render.githubusercontent.com/render/math?math={\mathscr{L}_1 = \left|2\sigma(D) - 1 - R\right|}">
<br>

```
loss = torch.abs(torch.sigmoid(delta) * 2 - 1 - result)
```

<img src="https://render.githubusercontent.com/render/math?math=\mathscr{L}_2 = -log(\sigma(R \times D)) %2B (1 - |R|) \times |D|">
<br>

```
loss = -torch.nn.functional.logsigmoid(result * delta) + (1 - torch.abs(result)) * torch.abs(delta)
```

(здесь будет график с функцией потерь)

Градиенты для всех микроматчей в турнире аккумулируется, после чего делается шаг градиентного спуска.


## Известные проблемы
(и возможные их решения)

1. У сигмоид исчезают градиенты при насыщении. Логарифм должен решить эту проблему, но это пока не тестировалось.
2. Алгоритм не моделирует отношения между игроками (возможно, кто-то лучше играет вместе). 
Решением может быть эмбеддинг второго уровня, но количество параметров от этого вырастет квадратично. 
Разреженный эмбеддинг может помочь, но это неточно.
3. Сумма/среднее не моделирует вклад отдельных игроков. Скорее всего, это сделать невозможно
4. Не учитываются редакторы/сложность. 

--------
[rating.chgk.info](rating.chgk.info) is a database of
 ["What? Where? When?" intellectual game](https://en.wikipedia.org/wiki/What%3F_Where%3F_When%3F) tournaments.