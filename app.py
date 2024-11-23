"""Web application."""
import time
from io import BytesIO

import streamlit as st
import torch
from PIL import Image
from scipy import ndimage
from streamlit_image_zoom import image_zoom

from utils.app_utils import IDX2LABEL, colorify_mask, blend_image_mask, pil_to_pt, segment_image, extract_layer, \
    generate_color_circle, ID2COLOR

st.set_page_config(page_title='Hepatocyte.ai', layout='wide')

@st.cache_resource
def load_model():
    return torch.jit.load('weights/manet-resnet101 (all classes)_scripted.pth')


def clear_st_session():
    st.session_state.clear()


model = load_model()


with st.container():
    _, image_load_column, _ = st.columns(3, vertical_alignment='center')
    with image_load_column:
        image_buffer = st.file_uploader(label='Загрузите изображение', type='png', on_change=clear_st_session())

with st.container():
    if image_buffer:
        pil_image = Image.open(image_buffer)
        pt_image = pil_to_pt(pil_image)
        mask = segment_image(model, pt_image)

        column_image, column_stats, _ = st.columns([0.5, 0.4, 0.1], gap='large')
        with column_image:
            if 'colored_mask' not in st.session_state:
                colored_mask = colorify_mask(mask)
                st.session_state['colored_mask'] = colored_mask
            else:
                colored_mask = st.session_state['colored_mask']
            image_with_mask = blend_image_mask(pil_image, colored_mask)
            show_mask = st.toggle('Показывать маску', value=True)
            if show_mask:
                image_zoom(image_with_mask, mode='scroll')
            else:
                image_zoom(pil_image, mode='scroll')
            res_byte = BytesIO()
            image_with_mask.save(res_byte, format='PNG')
            _, download_button_column, new_load_column, _ = st.columns([0.1, 0.4, 0.4, 0.1],
                                                                       vertical_alignment='center')
            with download_button_column:
                st.download_button('Скачать', res_byte, file_name='res.png', use_container_width=True)

        with column_stats:
            st.header('Морфометрия')
            labels = list(IDX2LABEL.items())[1:]
            for idx, name in list(IDX2LABEL.items())[1:]:
                label_mask = extract_layer(mask, idx)
                _, num_labels = ndimage.label(label_mask)
                c1, c2 = st.columns([0.05, 0.95], vertical_alignment='bottom')
                with c1:
                    st.html(generate_color_circle(ID2COLOR[idx].name.lower()))
                with c2:
                    st.write(f'{name}:', num_labels)
