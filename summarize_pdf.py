import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from models import *


# saves the rag file associated with each type of data in memory
def create_PDF_rag(text, file_name) -> str:
    rag_file = f"محتوى الملف النصى {file_name}:\n{text}\n\n"
    return rag_file


def extract_text_from_pdf(pdf_docs):
    documents_rag_file = ""
    rag_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            rag_text += page.extract_text()
        documents_rag_file += create_PDF_rag(rag_text, pdf.name)
        rag_text = ""
    return documents_rag_file


load_dotenv()
pdfs = st.file_uploader(
    "هنا" + " PDF " + "تحميل ملفات ال",
    type=["pdf"],
    accept_multiple_files=True,
)
max_out_tokens = st.slider("اقصى عدد الكلمات", 50, 1000, 200)

if st.button("ملخص النصوص"):
    rag_text = extract_text_from_pdf(pdfs)
    user_input = "ما هو ملخص الملفات النصية؟"
    if check_token_count(user_input, rag_text) <= 8000:  # type: ignore
        output = get_lama3_chat_response(
            user_input, rag_text, max_out_tokens  # type: ignore
        )
    else:
        output = get_gemini_pro_chat_response(rag_text, user_input, max_out_tokens)
    st.write(output)
