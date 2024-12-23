import streamlit as st
from scipy import ndimage
from streamlit_image_zoom import image_zoom

from utils.app_utils import buffer_to_pil, generate_color_circle
from utils.image_utils import segment_patch, extract_layer
from utils.model_utils import load_model

model = load_model(st.session_state['chosen_model'])

buffers = st.file_uploader(
    label='Загрузите изображение',
    type='png',
    accept_multiple_files=True
)


@st.fragment
def view_result_container(i, raw_img, mask, res_img):
    column_image, column_stats = st.columns(2, gap='large')
    with column_image:
        hide_mask = st.toggle('Скрыть маску', key=f'toggle-{i}')
        if hide_mask:
            image_zoom(raw_img, mode='both', keep_resolution=True)
        else:
            image_zoom(res_img, mode='both', keep_resolution=True)

    with column_stats:
        pass
        # labels = list(IDX2LABEL.items())[1:]
        # for idx, name in list(IDX2LABEL.items())[1:]:
        #     label_mask = extract_layer(mask, idx)
        #     _, num_labels = ndimage.label(label_mask)
        #     c1, c2 = st.columns([0.05, 0.95], vertical_alignment='bottom')
        #     with c1:
        #         st.html(generate_color_circle(ID2COLOR[idx].name.lower()))
        #     with c2:
        #         st.write(f'{name}:', num_labels)


if buffers:
    for idx, buffer in enumerate(buffers):
        pil_image = buffer_to_pil(buffer)
        segmented_patch = segment_patch(model, pil_image)
        view_result_container(idx, pil_image, segmented_patch['raw_predict'], segmented_patch['blended_predict'])
