import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from transformers import pipeline
import plotly.express as px
import requests
import nltk
import document_uploader as docup
import llm_helper as llmh
import sklearn_helper as sklh

st.set_page_config(page_title="Data mining with LLMs")

st.title("LLM usage for performing data mining tasks")
st.caption("by: Yaroslav Panasiuk")

# Sidebar for Task Selection
task = st.sidebar.selectbox(
    r"$\textsf{\large Select a task}$", 
    ["Summarization", "Classification", "Regression", "Clustering", "Correlation Analysis", "Visualization"]
)

# Sidebar to choose method: LLM or Data Mining
method = st.sidebar.radio(r"$\textsf{\large Select method:}$", ["Mistral Large", "LLama 3.2 8B", "Claude 3", "Standart Data Mining Techniques"])

# Main UI
st.write("## Chat with AI and Perform Tasks")

# File Upload Options
if "context_options" not in st.session_state:
    st.session_state.context_options = ["Upload a local file", "Use a link to a file", "Use a link to YouTube video"]
file_source = st.radio(r"$\textsf{\large Select file source:}$", st.session_state.context_options)
uploaded_file = None
file_link = None
youtube_link = None
user_question = None
target_column = None
support_column = None
cluster_count = 1

if file_source == "Upload a local file":
    uploaded_file = st.file_uploader(r"$\textsf{\large Upload a file}$", type=["csv", "txt", "pdf"], key="file1")
    docup.save_uploaded_file(uploaded_file)
    
elif file_source == "Use a link to a file":
    file_link = st.text_input(r"$\textsf{\large Enter the file link:}$")
    docup.save_file_by_url(file_link)
else: 
    youtube_link = st.text_input(r"$\textsf{\large Enter link to YouTube video:}$")

if task == "Summarization" and method != "Standart Data Mining Techniques" :
    st.session_state.context_options = ["Upload a local file", "Use a link to a file", "Use a link to YouTube video"]
else:
    st.session_state.context_options = ["Upload a local file", "Use a link to a file"]

if task == "Classification" :
    target_column = st.text_input(r"$\textsf{\large Enter the name of the column you want to target:}$")
    support_column = st.text_input(r"$\textsf{\large Enter the names of the columns to take the data from:}$")
    user_question = st.text_area(r"$\textsf{\large Ask your question or describe your task:}$", max_chars=10000)

elif task == "Clustering" and method != "Standart Data Mining Techniques":
    support_column = st.text_input(r"$\textsf{\large Enter the names of the columns to take the data from:}$")
    cluster_count = st.text_input(r"$\textsf{\large Enter the number of clusters:}$")

else:
    user_question = st.text_area(r"$\textsf{\large Ask your question or describe your task:}$", max_chars=10000)


# User Input (Optional)
if st.button("Submit"):
    st.write(f"Processing task: {task} using {method}")

# Process Tasks
if task == "Summarization":
    if method != "Standart Data Mining Techniques":
        if youtube_link:
            st.write("### Processing YouTube video...")
            response = llmh.summarize_youtube(youtube_link, user_question)
            st.write("## Summary:")
            st.write(response)
        elif uploaded_file or file_link:
            response = "unable to analyze"
            if uploaded_file:
                response = llmh.summarize_document(uploaded_file, user_question)
            if file_link:
                print(file_link)
            st.write("## Summary:")
            st.write(response)
        elif user_question:
            st.write("### Processing provided text...")
            response = llmh.summarize_text(user_question)
            st.write("## Summary:")
            st.write(response)
        else:
            st.warning("Please upload a file or provide a link to summarize.")
    else:  # Data Mining Techniques (Basic Text Processing)
        st.write("## Summarization using Basic Text Processing")
        if uploaded_file or file_link or user_question:
            response = sklh.summarize_text(document_name=uploaded_file.name, input=user_question)
            st.write("## Summary:")
            st.write(response)
        else:
            st.warning("Please upload a file or provide a link to summarize.")

elif task == "Classification":
    if method != "Standart Data Mining Techniques":
        st.write("### Classification using LLM")
        if uploaded_file or file_link:
            #df = pd.read_csv('output.csv')
            #fig = px.scatter(df, x='tsne_x', y='tsne_y', color='label', title="Sure! Here is Scatter Plot with classification of Symptom2Disease dataset:", labels={'color': 'Label'})
            #st.plotly_chart(fig, use_container_width=True)
            
            fig, metrics = llmh.classification(target_column_name=target_column, support_column_name=support_column, user_query=user_question)
            st.plotly_chart(fig, use_container_width=True)
            st.write(sklh.metrics_to_str(metrics))
            print(target_column)
            print(support_column)
            print(user_question)
            
        else:
            st.warning("Please upload a file or provide a link.")
    else:  # Data Mining Techniques (scikit-learn)
        st.write("### Classification using Data Mining")
        if uploaded_file or file_link:
            fig, metrics = sklh.classification(target_column_name=user_question)
            st.plotly_chart(fig, use_container_width=True)
            st.write(sklh.metrics_to_str(metrics))
        else:
            st.warning("Please upload a file or provide a link.")

elif task == "Regression":
    if method != "Standart Data Mining Techniques":
        st.write("### Regression using LLM")
        st.warning("LLM-based regression is not implemented yet. Please choose Data Mining Techniques.")
    else:  # Data Mining Techniques (scikit-learn)
        st.write("### Regression using Data Mining")
        if uploaded_file or file_link:
            fig, metrics = sklh.regression(target_column_name=user_question)
            st.plotly_chart(fig, use_container_width=True)
            st.write(sklh.metrics_to_str(metrics))
        else:
            st.warning("Please upload a file or provide a link.")

elif task == "Clustering":
    if method != "Standart Data Mining Techniques":
        st.write("### Clustering using LLM")
        if uploaded_file or file_link:
            fig, metrics = llmh.clustering(n_clusters=int(cluster_count))
            st.plotly_chart(fig, use_container_width=True)
            st.write(sklh.metrics_to_str(metrics))
        else:
            st.warning("Please upload a file or provide a link.")
    else: 
        st.write("### Clustering using Data Mining")
        if uploaded_file or file_link:
            fig, metrics = sklh.clustering(n_clusters=int(cluster_count))
            st.plotly_chart(fig, use_container_width=True)
            st.write(sklh.metrics_to_str(metrics))
        else:
            st.warning("Please upload a file or provide a link.")

elif task == "Correlation Analysis":
    if method != "Standart Data Mining Techniques":
        st.write("### Correlation Analysis using LLM")
        st.warning("LLM-based correlation analysis is not implemented yet. Please choose Data Mining Techniques.")
    else:  # Data Mining Techniques (Correlation Analysis)
        if uploaded_file or file_link:
            fig = sklh.correlation_analysis()
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please upload or provide links to both files.")

elif task == "Visualization":
    st.write("### Visualization Task")
    if uploaded_file or file_link:
        if file_link:
            df = pd.read_csv(file_link)
        else:
            df = pd.read_csv(uploaded_file)

        st.write("### Dataset:")
        st.write(df.head())

        # Simple visualization
        st.write("### Scatter Plot")
        x_col = st.selectbox("Select X-axis column:", df.columns)
        y_col = st.selectbox("Select Y-axis column:", df.columns)
        st.scatter_chart(df[[x_col, y_col]])
    else:
        st.warning("Please upload a file or provide a link.")    
    
    
# Footer with Example Links
links = {"Classification": """
            - [Social Network Ads Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Iris Flower Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Wine Quality Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Social Network Ads Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [MNIST Handwritten Digits Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv) (Image recognition not yet implemented)
         """, 
         "Summarization": """
            - [arXiv Summarization Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [News Articles Summarization Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [CNN/Daily Mail Summarization Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [XSum Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Scientific Papers Summarization Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
         """, 
         "Regression": """
            - [WHO Statistics on Life Expectancy](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Fish Market Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Red Wine Quality](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Medical Insurance Costs](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Real Estate Price Prediction](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
         """, 
         "Clustering": """
            - [Iris Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [KMTest Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Ecoli Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Wholesale Customers Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Wine Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
         """, 
         "Correlation Analysis": """
            - [Correlation of Global Health Metrics](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Education and Poverty Rates](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Climate Data Correlation](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Employment and Economic Indicators](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Sports Performance vs. Economic Indicators](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
         """, 
         "Visualization": """
            - [MTA Daily Ridership](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Halloween Candy Rankings](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [US Candy Distributor Sales Data](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [Bank Customer Churn](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
            - [BuzzFeed News Investigations](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
        """, 
         }
st.write("---")
st.write(f"#### Here are some links to .csv files to test {task} task")
st.write(links.get(task))
