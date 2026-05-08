# Лабораторная работа 2

## Цель

Реализованы два стохастических метода оптимизации поверх конструктивного числа: `SimulatedAnnealing` и `ParticleSwarm`. Конструктивное число задано интервалом `[left, right]`; оптимизаторы хранят точки как интервалы, а значение функции берется по середине интервала с сохранением текущей погрешности.

## Тестовые функции

В `src/benchmarks.py` добавлены функции из задания: Rastrigin, Ackley, Sphere, Rosenbrock, Beale, Goldstein-Price, Booth, Bukin N.6, Matyas, Levi N.13, Himmelblau, Three Hump Camel, Easom, Cross-in-tray, Eggholder, Holder table, McCormick, Schaffer N.2, Schaffer N.4 и функция из Desmos:

```text
0.047 * ((((x * (round(sin(10y)) + 2))^2 + y - 10)^2)
       + (x + (y * (round(sin(7x)) + 2))^2 - 7)^2)
```

## Теоретическое сравнение

`SimulatedAnnealing` делает один локальный случайный шаг за итерацию и иногда принимает ухудшение. Поэтому метод экономен по памяти и числу вызовов функции, но на многоэкстремальных и разрывных функциях может застревать около локальных минимумов после охлаждения.

`ParticleSwarm` хранит популяцию частиц и обменивает информацию через глобально лучшую найденную точку. Поэтому метод требует больше памяти и вызовов функции, но обычно быстрее покрывает сложный рельеф Rastrigin, Eggholder, Cross-in-tray, Holder table и функцию из Desmos.

## Запуск

```powershell
python run_experiments.py
```

Результаты сохраняются в `results/comparison.csv`, сводная таблица в `results/method_summary.md`, главный график ошибки значения в `results/value_error.png`.

Дополнительные графики сохраняются в `results/plots`:

- `distance_to_optimum.png` - расстояние до известного минимума;
- `function_calls.png` - количество вызовов функции;
- `elapsed_time.png` - время работы;
- `memory_usage.png` - оценка памяти;
- `error_heatmap.png` - тепловая карта ошибок;
- `method_wins.png` - количество функций, где метод оказался ближе к известному оптимуму.
