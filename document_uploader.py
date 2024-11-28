import pandas as pd
import numpy as np
import plotly.express as px
import nltk
import PyPDF2
import os
import pdfplumber
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from urllib.parse import urlparse
import mimetypes
import requests
import os



def text_from_document(file_name):
    if file_name is None or file_name == "":
        return
    ext = os.path.splitext(file_name)[1].lower() 
    if ext == ".txt":
        with open("./temp/current-file.txt", 'r', encoding='utf-8') as file:
            return file.read()
    elif ext == ".pdf":
        return extract_text_with_pdfplumber("./temp/current-file.pdf")
    else:
        return
    
    

def extract_text_with_pdfplumber(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


def save_uploaded_file(file):
    if file is None:
        return
    ext = os.path.splitext(file.name)[1].lower() 
    temp_file_path = os.path.join("temp", f"current-file{ext}")
    
    os.makedirs("temp", exist_ok=True)

    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())    
    return temp_file_path

def delete_file(file):
    if file is None or not os.path.exists(file):
        return    
    os.remove(file)


def save_file_by_url(url):
    if url == None or url == "":
        return
    ext = get_file_extension_from_url(url)
    if ext == None or ext == "":
        return
    response = requests.get(url, stream=True)  # Stream to handle large files
    if response.status_code == 200:
        save_path = f"./temp/current-file{ext}"
        print(save_path)
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):  # Write in chunks
                file.write(chunk)
        print(f"File saved successfully to {save_path}")
    else:
        print(f"Failed to download file. HTTP status code: {response.status_code}")

    
    
def get_file_extension_from_url(url):
    # Send a HEAD request to fetch headers without downloading the file
    response = requests.get(url)
    
    # Get the content type from the headers
    content_type = response.headers.get('Content-Type', '')
    guessed_type = mimetypes.guess_extension(content_type)
    return guessed_type



