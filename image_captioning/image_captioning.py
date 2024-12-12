import streamlit as st
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
import pickle


# preproccsing the image
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # type: ignore
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # type: ignore
    return img


# generates the caption for the image
def get_image_caption(image_path, encoder, decoder, tokenizer, max_length, units):
    hidden = tf.zeros((1, units))
    temp_input = tf.expand_dims(load_image(image_path), 0)

    features = encoder(temp_input)

    dec_input = tf.expand_dims([tokenizer.word_index["<start>"]], 0)
    result = []

    for i in range(max_length):
        predictions, hidden = decoder([dec_input, hidden, features])

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == "<end>":
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result


def extract_text_from_image(image_files):
    captions = []
    for image in image_files:
        uplaoded_file_name = f"./images/copy_{image.name}"
        with open(uplaoded_file_name, "wb") as f:
            f.write(image.getbuffer())
        # Load the encoder and decoder models
        print("Loading models...")
        encoder = load_model("./encoder.h5")
        decoder = load_model("./decoder.h5")
        print("Models loaded successfully")
        # Load the tokenizer from the file
        print("Loading tokenizer and max_length and units...")
        with open("tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)
        with open("max_length.txt", "r") as file:
            max_length = int(file.read())
        with open("units.txt", "r") as file:
            units = int(file.read())
        print("Tokenizer and max_length and units loaded successfully")
        image_path = uplaoded_file_name
        print("Generating caption...")
        caption = get_image_caption(
            image_path, encoder, decoder, tokenizer, max_length, units
        )
        if "<end>" in caption:
            caption.pop()
        caption = " ".join(caption)
        print("Generated Caption:", caption)
        captions.append(caption)
    return captions


load_dotenv()
images = st.file_uploader(
    "تحميل الصور",
    type=["jpg"],
    accept_multiple_files=True,
)

if st.button("محتوى الصور"):
    if images:
        captions = extract_text_from_image(images)
    for image, caption in zip(images, captions):  # type: ignore
        st.image(image)
        st.write(f"{image.name}: {caption}")
