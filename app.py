import streamlit as st
import os


app = st.navigation([
    st.Page('web_app/patch.py', title='Патчи'),
    st.Page('web_app/wsi.py', title='WSI')
], )

with st.sidebar:
    chosen_model = st.selectbox(
        label='Выберите модель',
        options=os.listdir('weights'))
    st.session_state['chosen_model'] = f'weights/{chosen_model}'

app.run()