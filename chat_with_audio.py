import streamlit as st
from dotenv import load_dotenv
from models import *
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    BitsAndBytesConfig,
)
from streamlit_mic_recorder import speech_to_text


# saves the rag file associated with each type of data in memory
def create_audio_file(text, file_name) -> str:
    rag_file = f"محتوى الملف الصوتي {file_name}:\n{text}\n\n"
    return rag_file


def extract_text_from_audio(audio_files):
    output = ""
    for audio in audio_files:
        uplaoded_file_name = f"./audio/copy_{audio.name}"
        with open(uplaoded_file_name, "wb") as f:
            f.write(audio.getbuffer())
        torch_dtype = torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        model_id = "./whisper-large-v3"
        print(f"Loading quantized model {model_id} with dtype {torch_dtype}")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map="auto",
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        print(f"Loaded model {model_id}")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
        )
        print(f"Transcribing {uplaoded_file_name}")
        result = pipe(uplaoded_file_name, generate_kwargs={"language": "arabic"})
        print(f"Transcribing done. Result: {result['text']}")  # type: ignore
        output += create_audio_file(result["text"], audio.name)  # type: ignore
        os.remove(uplaoded_file_name)
    return output


load_dotenv()
audio = st.file_uploader(
    "تحميل ملفات الصوتية هنا",
    type=["wav", "mp3"],
    accept_multiple_files=True,
)
max_out_tokens = st.slider("اقصى عدد الكلمات", 50, 1000, 200)
text = speech_to_text(
    language="ar", start_prompt="Start recording", stop_prompt="Stop recording"
)
user_input = st.text_input(":question:" + "السؤال هنا", value=text if text else "")
if text != None and text != "":
    user_input = text

if st.button("أجب على السؤال"):
    if user_input and audio:
        rag_text = extract_text_from_audio(audio)
        if check_token_count(user_input, rag_text) <= 8000:  # type: ignore
            output = get_lama3_chat_response(
                user_input, rag_text, max_out_tokens  # type: ignore
            )
        else:
            output = get_gemini_pro_chat_response(rag_text, user_input, max_out_tokens)
        st.write(output)
