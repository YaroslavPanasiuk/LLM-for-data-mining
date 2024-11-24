from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import time

# Load data
loader = TextLoader("essay.txt")
docs = loader.load()
# Split text into chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
# Define the embedding model
embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key="gyjF9KXLwyquRqkmKXiQ2rjJoIwdjx6S")
# Create the vector store 
vector = FAISS.from_documents(documents, embeddings)
# Define a retriever interface
retriever = vector.as_retriever()
# Define LLM
model = ChatMistralAI(mistral_api_key="gyjF9KXLwyquRqkmKXiQ2rjJoIwdjx6S")
# Define prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that that can answer questions about content of uploaded document with {ext} extension
Answer the following question by searching the following document as a context:

<context>
{context}
</context>

Only use the factual information from the provided document to answer the question.
        
If you feel like you don't have enough information to answer the question, say "I don't know".
        
Your answers should be verbose and detailed.

Question: {input}""")

# Create a retrieval chain to answer questions
document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
ext = ".txt"
query = "Summarize the story"
time.sleep(3)
response = retrieval_chain.invoke({"input": "Summarize the story", "ext": ext})
print(response["answer"])
