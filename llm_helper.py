from langchain_community.document_loaders import YoutubeLoader, TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import LLMChain
from dotenv import load_dotenv
import time
import os


load_dotenv()
embeddings = MistralAIEmbeddings(model="mistral-embed")


def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    counter = 0
    
    while True:
        counter += 1
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
            docs = text_splitter.split_documents(transcript)
            db = FAISS.from_documents(docs, embeddings)
            break
        except:  
            if counter > 10:
                raise
    time.sleep(7)
    return db


def get_response_from_query(db, prompt, k=8):
    time.sleep(3)
    docs = db.similarity_search(prompt, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatMistralAI(model_name="mistral-large-latest")

    chain = prompt | llm

    response = chain.invoke({"question":prompt, "docs":docs_page_content})
    return response.content


def analyze_yt_by_url(url, question=None):
    query = question
    if question == None or question == "":
        query = "summarize whole video for me"
    db = create_db_from_youtube_video_url(url)
    
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template=f"""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {query}
        By searching the following video transcript: {db}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )
    
    return get_response_from_query(db, prompt)

def summarize_text(text, max_chars=10000):
    llm = ChatMistralAI(model_name="mistral-large-latest")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can summarize text up to {max_chars} characters in a few sentences
        
        Summaeize the following text: {text}
        
        Only use the most important information from the provided text answer the question.
        
        If you feel like you don't have enough information or context to summarize, say "I cannot summarize provided text".
        
        Your answers should be short and to the point.
        """,
    )
    chain = prompt | llm

    response = chain.invoke({"text":text, "max_chars":max_chars})
    return response.content
    
def create_db_from_document(uploaded_document, chunk_size=4000):
    document = save_uploaded_file(uploaded_document)
    ext = os.path.splitext(document)[1].lower()
    if ext == ".txt":
        loader = TextLoader(document, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(document)
    elif ext == ".csv":
        loader = CSVLoader(document)
    else:
        raise ValueError(f"Unsupported file type: {ext}")        
    docs = loader.load()
    counter = 0
    while True:
        counter += 1
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=70, length_function=len)
            documents = text_splitter.split_documents(docs)
            db = FAISS.from_documents(documents, embeddings)
            break
        except:  
            if counter > 10:
                raise
        finally:
            delete_file(document)
    time.sleep(7)
    return db  


def analyze_document_attachment(document, question=None):
    query = question
    ext = os.path.splitext(document.name)[1].lower() 
    if question == None or question == "":
        query = "summarize document for me"
    db = create_db_from_document(document)
    retriever = db.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant that that can answer questions about content of uploaded document with {ext} extension
    Answer the following question by searching the following document as a context:

    <context>
    {context}
    </context>

    Only use the factual information from the provided document to answer the question.
            
    If you feel like you don't have enough information to answer the question, say "I don't know".
            
    Your answers should be verbose and detailed. If user asks you to summarize the document then try to be laconic

    Question: {input}""")
    llm = ChatMistralAI(model_name="mistral-large-latest")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    time.sleep(7)
    response = retrieval_chain.invoke({"input": query, "ext": ext})
    
    
    return response.get("answer")


def save_uploaded_file(file):
    if file is None:
        return
    temp_file_path = os.path.join("temp", file.name)
    
    os.makedirs("temp", exist_ok=True)

    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())
    
    return temp_file_path

def delete_file(file):
    if file is None or not os.path.exists(file):
        return    
    os.remove(file)
    
    
def get_file_extension_from_url(url):
    # Send a HEAD request to fetch headers without downloading the file
    response = requests.head(url)
    
    # Get the content type from the headers
    content_type = response.headers.get('Content-Type', '')
    
    # Map content type to file extension
    if 'pdf' in content_type:
        return '.pdf'
    elif 'text' in content_type:
        return '.txt'
    elif 'csv' in content_type:
        return '.csv'
    # Add more content types as needed
    else:
        return None
