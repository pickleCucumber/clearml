# Оценка кредитного дефолта для новых клиентов Credit Line

Предсказания просрочки 60 дней+ для заявок новых клиентов, одобренных риск-моделью.

## Структура проекта
```
├── README.md
├── data
|   ├── raw
|   └── prepared
├── models
|   └── cl_lgbm_model.pkl
├── notebooks
|   ├── EDA DataShift.ipynb
|   ├── EDA NBKI.ipynb
|   |── EDA equifax.ipynb
|   ├── EDA megafon.ipynb
|   ├── EDA origin.ipynb
|   ├── EDA.ipynb
|   ├── EDA megafon.ipynb
|   ├── model LightGBM with new target 1 month out-of-sample.ipynb
|   ├── model LightGBM with new target 2 months out-of-sample.ipynb
|   ├── model LightGBM with new target.ipynb
|   ├── model LightGBM with pca.ipynb
|   └── model LightGBM.ipynb
├── references
├── reports
|   └── ?
├── src
├── utils
└── mac_requirements.txt
```

## Интерпретатор и окружение
Я использовал интерпретатор [Python 3.11.4](https://www.python.org/downloads/release/python-3114/).

Рекомендую использовать виртуальное окружение. Его можно создать следующей командой:
```
python -m venv venv
```
Все необходимые пакеты собраны в файле `requirements.txt`.
Перед установкой пакетов не забудьте активировать виртуальную среду.
Вы можете установить пакеты одной командой:
```
pip install -r requirements.txt
```

После установки пакетов необходимо добавить текущий проект как пакет
С помощью команды:
```
pip install -e .
```

## Работа с файлами .env для безопасного хранения конфигураций

Для управления конфиденциальной информацией и переменными окружения, наш проект использует файлы `.env`. Эти файлы позволяют нам отделять конфигурационные данные от исходного кода, что повышает безопасность и гибкость нашего приложения.

Пример конфига для этого проекта можно посмотреть в .env_example


## Hardware

Linux, процессор - 13th Gen Intel(R) Core(TM) i9-13900 | 16 ядер
