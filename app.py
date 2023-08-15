import streamlit as st
import torch
from template import css, bot_template, user_template
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatGooglePalm
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "BAAI/bge-base-en")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatGooglePalm()
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory,
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response)

def main():
    load_dotenv()
    st.set_page_config(page_title="pdfGPT", page_icon=":books:", layout="wide")

    st.write(css, unsafe_allow_html=True)


    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("pdfGPT :books:")
    user_question = st.text_input("Enter your question here")
    if user_question:
        handle_userinput(user_question)
    st.write(user_template.replace("{{MSG}}", "Hello Bot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "hello Human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your pdf file here", type="pdf", accept_multiple_files=True, key="pdf")
        if st.button("Upload"):
            with st.spinner("Uploading..."):
                
                #get the uploaded file
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                #get the text chunk from the pdf
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                
                #create vector store
                vector_store = get_vector_store(text_chunks)
                # st.write(vector_store)

                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)
    
if __name__ == "__main__":
    # print(torch.cuda.is_available())
    main()