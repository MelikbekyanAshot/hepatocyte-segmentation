from enum import Enum
from typing import Tuple


class Color(Enum):
    TRANSPARENT = (0, 0, 0, 256)
    RED = (255, 0, 0, 128)
    GREEN = (0, 255, 0, 128)
    BLUE = (0, 0, 255, 128)
    YELLOW = (255, 255, 0, 128)
    MAGENTA = (255, 0, 255, 128)
    CYAN = (0, 255, 255, 128)
    BLACK = (0, 0, 0, 128)


class ColorManager:
    """
    Класс для управления цветами и их соответствием типам клеток.
    """
    def __init__(self):
        self.idx2label = {
            0: '_background_',
            1: 'Баллонная дистрофия',
            2: 'Инклюзионные',
            3: 'Безъядерные',
            4: 'Нормальные',
            5: 'Жировые',
            6: 'Мезенхимальные',
            7: 'Безъядерные жировые'
        }
        self.colors = [Color.TRANSPARENT, Color.RED, Color.GREEN, Color.BLUE,
                       Color.YELLOW, Color.MAGENTA, Color.CYAN, Color.BLACK]

    @property
    def cell_groups(self):
        return [group for idx, group in list(self.idx2label.items())[1:]]

    def __getitem__(self, index: int) -> Tuple[Tuple[int, int, int, int], str]:
        """
        Возвращает цвет и метку класса по индексу.

        Args:
            index: Индекс цвета.

        Returns:
            Tuple[Tuple[int, int, int, int], str]: Кортеж из цвета (RGBA) и метки класса.

        Raises:
            IndexError: Если индекс вне допустимого диапазона.
        """
        if 0 <= index < len(self.colors):
            return self.colors[index], self.idx2label[index]
        else:
            raise IndexError(f"Индекс {index} вне диапазона допустимых значений.")

    def get_color_by_label(self, label: str) -> Tuple[int, int, int, int]:
        """
        Возвращает цвет по метке класса.

        Args:
            label: Метка класса.

        Returns:
            Tuple[int, int, int, int]: Цвет (RGBA).

        Raises:
            KeyError: Если метка класса не найдена.
        """
        for idx, lbl in self.idx2label.items():
            if lbl == label:
                return self.colors[idx].value
        raise KeyError(f"Метка класса '{label}' не найдена.")
