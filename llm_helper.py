from langchain_community.document_loaders import YoutubeLoader, TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from urllib.parse import urlparse
import mimetypes
import pandas as pd
import document_uploader as docup
import sklearn_helper as sklh
from mistralai import Mistral
import requests
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


def create_db_from_document(uploaded_document, chunk_size=4000):
    document = docup.save_uploaded_file(uploaded_document)
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
            docup.delete_file(document)
    time.sleep(7)
    return db  


def summarize_youtube(url, query=None, k=2):
    if query is None or query == "":
        question = "Shortly sumarize the video"
    else:
        question = f"Answer the following question: {query}"
    db = create_db_from_youtube_video_url(url)
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatMistralAI(model_name="mistral-large-latest")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = prompt | llm
    time.sleep(5)
    response = chain.invoke({"question":question, "docs": docs_page_content})
    return response.content


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
    

def summarize_document(document, question=None):
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


def get_embeddings_by_chunks(data, chunk_size):
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    model = "mistral-embed"
    chunks = [data[x : x + chunk_size] for x in range(0, len(data), chunk_size)]
    embeddings_response = []
    for c in chunks:
        time.sleep(2)
        embeddings_response.append(client.embeddings.create(model=model, inputs=c))
        print("doing embeddings")
    return [d.embedding for e in embeddings_response for d in e.data]


def get_text_embedding(inputs):
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    embeddings_batch_response = client.embeddings.create(
        model = "mistral-embed",
        inputs=inputs
    )
    return embeddings_batch_response.data[0].embedding


def classification(target_column_name=None, support_column_name=None, user_query=None):
    if support_column_name == None or support_column_name == "":
        support_column_name = "text"
    
    df = pd.read_csv("temp/current-file.csv")
    df.drop(columns=['embeddings'])
    df['embeddings'] = get_embeddings_by_chunks(df[support_column_name].tolist(), 50)
    df.to_csv("temp/current-file.csv")
    return sklh.classification(target_column_name, support_column_names='embeddings', user_query=user_query, dataframe=df)
    
    
def clustering(n_clusters=7, support_column_name="text"):    
    df = pd.read_csv("temp/current-file.csv")
    df.drop(columns=['embeddings'])
    df['embeddings'] = get_embeddings_by_chunks(df[support_column_name].tolist(), 50)
    df.to_csv("temp/current-file.csv")
    return sklh.clustering(n_clusters=n_clusters, dataframe=df)
    