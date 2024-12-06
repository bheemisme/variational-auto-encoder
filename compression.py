import streamlit as st
import os
import pathlib
from pathlib import Path
from model.model import compress, preprocess_image
from PIL import Image

def compression():
    st.markdown('### Compression')
    input_images_dir = Path('images')

    uploaded_file = st.file_uploader("Choose file: ", accept_multiple_files=False)

    if uploaded_file:

        if uploaded_file.type == "image/jpeg" or uploaded_file.type == "image/png":
            input_image_path = pathlib.Path(input_images_dir / uploaded_file.name)
            input_image_path.touch()
            with open(input_image_path, "wb") as input_file:
                input_file.write(uploaded_file.read())
            
            img = preprocess_image(image_path=input_image_path)
            cmp_path = compress(img, uploaded_file.name)

            
            image = Image.open(input_image_path)
            st.image(image=image, caption="Input")

            @st.fragment
            def download():
                with open(cmp_path, "rb") as file:
                    
                    btn = st.download_button(
                        label="Download compressed image",
                        data=file,
                        file_name=Path(cmp_path).name,
                        mime="application/octet-stream",
                    )
            download()

