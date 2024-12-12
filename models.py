import subprocess
import re
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os


# a function to check the number of tokens in rag file
def check_token_count(user_input, rag_file):
    lama3_prompt = f"""نسخة من مربع حوار، حيث يتفاعل المستخدم مع مساعد يُدعى بوب. بوب مفيد، ولطيف، وصادق، وجيد في الكتابة، ولا يفشل أبدًا في الإجابة على طلبات المستخدم على الفور وبدقة إذا لم يكن الجواب موجودا
في السياق المقدم، فقط قل "لا أعرف الاجابة"،
لا تقدم إجابة خاطئة واختم اجابتك بنقطة.

{rag_file}المستخدم: {user_input}
بوب:"""
    with open("./rag_prompt/rag_file.txt", "w") as f:
        f.write(lama3_prompt)
    output = subprocess.check_output(
        '../llama.cpp/tokenize ./arabic-orpo-llama-3-8b-instruct.Q4_K_M.gguf "$(cat ./rag_prompt/rag_file.txt)"',
        shell=True,
    )
    output = output.decode("utf-8", "ignore")
    tokens = output.count(" -> ")
    return tokens


# a function to get the response from lama3 model
def get_lama3_chat_response(user_input, rag_file, max_out_tokens):
    lama3_prompt = f"""نسخة من مربع حوار، حيث يتفاعل المستخدم مع مساعد يُدعى بوب. بوب مفيد، ولطيف، وصادق، وجيد في الكتابة، ولا يفشل أبدًا في الإجابة على طلبات المستخدم على الفور وبدقة اختم اجابتك بنقطة.

{rag_file}المستخدم: {user_input}
بوب:"""
    with open("./rag_prompt/rag_file.txt", "w") as f:
        f.write(lama3_prompt)
    command = f'../llama.cpp/main -m ./arabic-orpo-llama-3-8b-instruct.Q4_K_M.gguf -n {max_out_tokens} -e --ctx-size 8000 --repeat_penalty 1.0 --temp 0.8 --color -r "المستخدم:" -f ./rag_prompt/rag_file.txt --in-prefix " " --in-suffix "بوب:" --log-disable'
    output = subprocess.check_output(command, shell=True)
    output_str = output.decode("utf-8", "ignore")
    # Define the pattern to match special tokens
    pattern = r"<\|.*?\|>"
    # Use re.sub() to replace the matched patterns with an empty string
    output_str = re.sub(pattern, "", output_str)
    output_str = output_str.split("بوب:")[-1]
    remove_index = output_str.find("المستخدم:")
    if remove_index != -1:
        output_str = output_str[:remove_index]
    return output_str


# a function to get the response from gemini-pro model
def get_gemini_pro_chat_response(context, user_question, max_out_tokens) -> str:
    # Retrieve API key from environment variable
    google_api_key = os.getenv("GOOGLE_API_KEY")
    # Split Texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    texts = text_splitter.split_text(context)
    # Chroma Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key
    )  # type: ignore
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever()
    # Get Relevant Documents
    docs = vector_index.get_relevant_documents(user_question)
    # Define Prompt Template
    prompt_template = """أجب عن السؤال بشكل مفصل قدر الإمكان من السياق المقدم، وتأكد من تقديم جميع التفاصيل، إذا لم تكن الإجابة في السياق المقدم فقط قل، "لا اعرف الاجابة"،
لا تقدم إجابة خاطئة.\n\n
السياق:\n {context}\n
السؤال: \n{question}\n
إلاجابة:
"""
    # Create Prompt
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    temp = 0.8
    top_k = 40
    top_p = 0.9
    # Load QA Chain
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=google_api_key,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        max_output_tokens=max_out_tokens,
    )  # type: ignore
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    # Get Response
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )
    return response["output_text"]


def get_image_chat_response(image_path, user_input, max_out_tokens):
    command = f'../llama.cpp/llava-cli -m ./llava-llama-3-8b-v1_1-int4.gguf --mmproj ./llava-llama-3-8b-v1_1-mmproj-f16.gguf --image {image_path} --temp 0.1 -n {max_out_tokens} -c 4096 -e -p "<|start_header_id|>user<|end_header_id|>\n\n<image>\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" --log-disable'
    output = subprocess.check_output(command, shell=True)
    output_str = output.decode("utf-8", "ignore")
    return output_str
