import streamlit as st
from streamlit_image_zoom import image_zoom

from models.pipeline import StreamlitPipeline
from utils.app_utils import buffer_to_pil, pil_to_np
from utils.model_utils import load_model

model = load_model(st.session_state['chosen_model'])
pipe = StreamlitPipeline(model)

buffer = st.file_uploader(
    label='Загрузите изображение',
    type='png'
)

if buffer:
    wsi = buffer_to_pil(buffer).resize((8192, 8192))
    result = pipe.run(wsi)
    image_zoom(
        result,
        mode='dragmove',
        keep_resolution=True,
        zoom_factor=10
    )


