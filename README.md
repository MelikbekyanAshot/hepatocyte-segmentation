# О проекте

# Данные

# Запуск
Параметры обучения модели хранятся в файле [конфигурации](config.yml).
В файле конфигурации можно выбрать:
- общую архитектуру модели
- архитектуру внутренних блоков модели
- размер батча
- количество эпох обучения
- оптимизатор
- функцию потерь

Процесс обучения логируется на [wandb](https://wandb.ai/melikbekyan-ashot/hepatocyte-segmentation).

Для запуска процесса обучения с выбранными параметрами необходимо 
запустить файл `train.py` через консоль или IDE.

# Используемые технологии
- [PyTorch Lightning](https://pytorch-lighting.readthedocs.io/en/latest/) - фреймфорк для обучения модели.
- [Segmentation Models](https://pypi.org/project/segmentation-models-pytorch/) - конструктор моделей для сегментации.
- [Albumentations](https://albumentations.ai/docs/examples/pytorch_classification/) - аугментация данных.
- [wandb](https://wandb.ai/home) - платформа для логирования экспериментов и анализа данных.