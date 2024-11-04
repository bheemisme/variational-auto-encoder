import streamlit as st

from pathlib import Path
from model.model import decompress
from PIL import Image
import matplotlib.pyplot as plt
import torch

def decompression():
    st.markdown('### Decompression')

    uploaded_file = st.file_uploader("Choose file: ", accept_multiple_files=False, key='decompression')

    if uploaded_file:
        if uploaded_file.type == 'application/octet-stream':
            img_path = decompress(uploaded_file)
            img = Image.open(img_path)

            st.image(img)

            @st.fragment
            def download():
                with open(img_path, "rb") as file:
                    
                    btn = st.download_button(
                        label="Download image",
                        data=file,
                        file_name=Path(img_path).name,
                        mime="application/octet-stream",
                        key='decomp-image'
                    )
            download()
            
