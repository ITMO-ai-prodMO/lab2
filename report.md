# Отчет по лабораторной работе №2  
## Методы оптимизации и конструктивные числа

**Авторы:**  
Голубева Мария Сергеевна, ИСУ: 465572  
Стринадкина Полина Николаевна, ИСУ: 467611  
Группа J3213

---

# Цель работы

Изучение стохастических методов безусловной оптимизации и сравнение их эффективности с детерминированными методами на различных тестовых функциях.

---

# Задачи работы

1. Реализовать методы:
   - Random Search
   - Simulated Annealing

2. Провести сравнение с методом:
   - Nelder–Mead

3. Выполнить тестирование на функциях:
   - Sphere
   - Rosenbrock
   - Rastrigin
   - Himmelblau
   - Desmos discontinuous function

4. Сравнить методы по:
   - скорости;
   - памяти;
   - числу вызовов;
   - качеству решения.

---

# Теоретическая часть

## Random Search

Метод случайного поиска основан на генерации случайных точек в области поиска и выборе лучшей найденной.

## Simulated Annealing

Метод имитации отжига допускает ухудшающие переходы, что позволяет выходить из локальных минимумов.

## Nelder–Mead

Симплексный детерминированный метод без вычисления производных.

---

# Результаты экспериментов

# Количество вызовов функции

![](img/calls_Sphere_function_2_variables.png)

![](img/calls_Rosenbrock_function_2_variables.png)

![](img/calls_Rastrigin_function_2_variables.png)

![](img/calls_Himmelblau_function.png)

![](img/calls_Desmos_discontinuous_function.png)

### Вывод

Nelder–Mead требует значительно меньше вызовов функции.

---

# Сходимость методов

![](img/convergence_Sphere_function_2_variables.png)

![](img/convergence_Rosenbrock_function_2_variables.png)

![](img/convergence_Rastrigin_function_2_variables.png)

![](img/convergence_Himmelblau_function.png)

![](img/convergence_Desmos_discontinuous_function.png)

### Вывод

Nelder–Mead быстрее сходится на гладких функциях.  
Simulated Annealing устойчивее к локальным минимумам.

---

# Использование памяти

![](img/memory_Sphere_function_2_variables.png)

![](img/memory_Rosenbrock_function_2_variables.png)

![](img/memory_Rastrigin_function_2_variables.png)

![](img/memory_Himmelblau_function.png)

![](img/memory_Desmos_discontinuous_function.png)

### Вывод

Nelder–Mead использует минимум памяти.  
Стохастические методы требуют больше ресурсов.

---

# Время работы

![](img/time_Sphere_function_2_variables.png)

![](img/time_Rosenbrock_function_2_variables.png)

![](img/time_Rastrigin_function_2_variables.png)

![](img/time_Himmelblau_function.png)

![](img/time_Desmos_discontinuous_function.png)

### Вывод

На большинстве тестов Nelder–Mead оказался самым быстрым методом.

---

# Общий вывод

В ходе лабораторной работы были исследованы стохастические методы оптимизации.

Установлено:

- **Nelder–Mead** — лучший на гладких функциях;
- **Random Search** — простой, но медленный;
- **Simulated Annealing** — эффективен на сложных многомодальных функциях.

Выбор метода зависит от структуры задачи оптимизации.

---