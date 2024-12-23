"""Utility components for web application."""
import numpy as np
from PIL import Image


def generate_color_circle(color: str):
    html_code = f"""
<head>
    <style>
        .red-circle {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #ff0000;
        }}
        .green-circle {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #00ff00;
        }}
        .blue-circle {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #0000ff;
        }}
        .yellow-circle {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #ffff00;
        }}
        .cyan-circle {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #00ffff;
        }}
        .magenta-circle {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #ff00ff;
        }}
        .black-circle {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #000;
        }}
    </style>
</head>
<body>
    <span class="{color}-circle"></span>
</body>
</html>"""
    return html_code


def buffer_to_pil(buffer):
    image = Image.open(buffer)
    return image


def pil_to_np(pil_image: Image) -> np.ndarray:
    return np.array(pil_image)