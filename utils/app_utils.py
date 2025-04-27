"""Utility components for web application."""
import numpy as np
from PIL import Image


def generate_color_circle(color: str):
    html_code = f"""
<head>
    <style>
        .circle-1 {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #ff0000; /* red */
        }}
        .circle-2 {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #00ff00; /* green */
        }}
        .circle-3 {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #0000ff; /* blue */
        }}
        .circle-4 {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #ffff00; /* yellow */
        }}
        .circle-5 {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #ff00ff; /* magenta */
        }}
        .circle-6 {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #00ffff; /* cyan */
        }}
        .circle-7 {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #000; /* black */ 
        }}
    </style>
</head>
<body>
    <span class="circle-{color}"></span>
</body>
</html>"""
    return html_code


def buffer_to_pil(buffer):
    image = Image.open(buffer)
    return image


def pil_to_np(pil_image: Image) -> np.ndarray:
    return np.array(pil_image)