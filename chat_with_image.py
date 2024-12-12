import streamlit as st
from dotenv import load_dotenv
from translation import *
from models import *
from streamlit_mic_recorder import speech_to_text

load_dotenv()
image = st.file_uploader(
    "تحميل الصورة",
    type=["jpg"],
    accept_multiple_files=False,
)
max_out_tokens = st.slider("اقصى عدد للكلمات", 50, 1000, 200)
text = speech_to_text(
    language="ar", start_prompt="Start recording", stop_prompt="Stop recording"
)
user_input = st.text_input(":question:" + "السؤال هنا", value=text if text else "")
if text != None and text != "":
    user_input = text

if st.button("أجب على السؤال"):
    if image and user_input:
        uplaoded_file_name = f"./images/copy_{image.name}"
        with open(uplaoded_file_name, "wb") as f:
            f.write(image.getbuffer())
        user_input = translate_to_english(user_input)
        output = get_image_chat_response(uplaoded_file_name, user_input, max_out_tokens)
        output = translate_to_arabic(output)
        st.image(image)
        st.write(f"{image.name}: {output}")
