import os
import glob
from flask import Flask, render_template, request, redirect, url_for, flash, session
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
from io import BytesIO
from dotenv import load_dotenv
from multiprocessing import Pool
from constants import CHROMA_SETTINGS
import tempfile
from tqdm import tqdm
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
import logging
import uuid
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
chunk_size = 500
chunk_overlap = 50
persist_directory = os.getenv('PERSIST_DIRECTORY')
source_directory = os.getenv('SOURCE_DIRECTORY', 'source_documents')
target_source_chunks = int(os.getenv('TARGET_SOURCE_CHUNKS', 5))
embeddings_model_name = os.getenv('EMBEDDINGS_MODEL_NAME')
model_type = os.getenv('MODEL_TYPE')
groq_api_key = os.getenv('groq_api_key')
google_api_key = os.getenv('GOOGLE_API_KEY')

# In-memory store for conversation objects
conversation_store = {}

# HTML CSS
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.pinimg.com/originals/0c/67/5a/0c675a8e1061478d2b7b21b330093444.gif" style="max-height: 70px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://th.bing.com/th/id/OIP.uDqZFTOXkEWF9PPDHLCntAHaHa?pid=ImgDet&rs=1" style="max-height: 80px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".pdf": (PyMuPDFLoader, {})
}

def load_single_document(file_content: BytesIO, file_type: str) -> List[Document]:
    ext = "." + file_type.rsplit("/", 1)[1]

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
        temp_file.write(file_content.getvalue())
        temp_file_path = temp_file.name

    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(temp_file_path, **loader_args)
        results = loader.load()
        os.remove(temp_file_path)
        return results

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_uploaded_documents(uploaded_files, uploaded_files_type, ignored_files: List[str] = []) -> List[Document]:
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(uploaded_files), desc='Loading new documents', ncols=80) as progress_bar:
            for i, uploaded_file in enumerate(uploaded_files):
                file_type = uploaded_files_type[i]
                file_content = BytesIO(uploaded_file.read())
                docs = load_single_document(file_content, file_type)
                results.extend(docs)
                progress_bar.update()
    return results

def get_pdf_text(uploaded_files):
    ignored_files = []  # Add files to ignore if needed

    uploaded_files_list = [file for file in uploaded_files]
    uploaded_files_type = [file.mimetype for file in uploaded_files]
    results = load_uploaded_documents(uploaded_files_list, uploaded_files_type, ignored_files)
    return results

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(
                os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 1 documents are needed in a working vectorstore
            if len(list_index_files) > 0:
                logger.info("Vectorstore exists")
                return True
    return False

def get_text_chunks(results, chunk_size, chunk_overlap):
    texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(results)
    return texts

def get_vectorstore(results, embeddings_model_name, persist_directory, client_settings, chunk_size, chunk_overlap):
    embeddings = CohereEmbeddings()
    logger.info('Cohere embeddings loaded')

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        logger.info(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = get_text_chunks(results, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if len(texts) > 0:
            db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        logger.info("Creating new vectorstore")
        logger.info("Creating embeddings. May take some minutes...")
        texts = get_text_chunks(results, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
        db.add_documents(texts)

    return db

def get_conversation_chain(vectorstore, target_source_chunks, model_type):
    retriever = vectorstore.as_retriever(search_kwargs={"k": target_source_chunks})
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_type)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    conversation_id = session.get('conversation_id')
    if conversation_id and conversation_id in conversation_store:
        conversation = conversation_store[conversation_id]
        response = conversation({'question': user_question})

        chat_history = []
        chat_history_html = ""

        for message in response['chat_history']:
            if isinstance(message, HumanMessage):
                chat_history.append({'role': 'user', 'content': message.content})
                chat_history_html += user_template.replace("{{MSG}}", message.content)
            elif isinstance(message, AIMessage):
                chat_history.append({'role': 'bot', 'content': message.content})
                chat_history_html += bot_template.replace("{{MSG}}", message.content)
            else:
                # Handle other types of messages if needed
                chat_history_html += bot_template.replace("{{MSG}}", str(message))

        # Store only serializable data in session
        session['chat_history'] = chat_history

        return chat_history_html
    else:
        flash('No active conversation. Please upload documents first.')
        return ""
