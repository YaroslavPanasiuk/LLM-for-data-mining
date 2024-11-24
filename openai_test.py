import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from transformers import pipeline
import requests
import llm_helper as llmh

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

if file_source == "Upload a local file":
    uploaded_file = st.file_uploader(r"$\textsf{\large Upload a file}$", type=["csv", "txt", "pdf"], key="file1")
elif file_source == "Use a link to a file":
    file_link = st.text_input(r"$\textsf{\large nter the file link:}$")
else: 
    youtube_link = st.text_input(r"$\textsf{\large Enter link to YouTube video:}$")

if task == "Summarization" and method != "Standart Data Mining Techniques" :
    st.session_state.context_options = ["Upload a local file", "Use a link to a file", "Use a link to YouTube video"]
else:
    st.session_state.context_options = ["Upload a local file", "Use a link to a file"]

# Special case for Correlation Analysis (requires two files)
if task == "Correlation Analysis":
    st.write("# Upload or provide links to **two files** for correlation analysis.")
    uploaded_file_2 = st.file_uploader(r"$\textsf{\large Upload the second file}$", type=["csv"], key="file2")
    file_link_2 = st.text_input(r"$\textsf{\large Enter the link to the second file}$", key="link2")

# User Input (Optional)
user_question = st.text_area(r"$\textsf{\large Ask your question or describe your task:}$", max_chars=10000)

if st.button("Submit"):
    st.write(f"Processing task: {task} using {method}")
    

# Process Tasks
if task == "Summarization":
    if method != "Standart Data Mining Techniques":
        if youtube_link:
            st.write("### Processing YouTube video...")
            response = llmh.analyze_yt_by_url(youtube_link, user_question)
            st.write("## Summary:")
            st.write(response)
        elif uploaded_file or file_link:
            response = "unable to analyze"
            if uploaded_file:
                response = llmh.analyze_document_attachment(uploaded_file, user_question)
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
        if uploaded_file or file_link:
            if file_link:
                file_content = requests.get(file_link).text
            else:
                file_content = uploaded_file.read().decode("utf-8")

            # Simple word-based summarization (example)
            sentences = file_content.split(".")
            summary = " ".join(sentences[:3])  # First 3 sentences as a summary
            st.write("## Summary:")
            st.write(summary)
        else:
            st.warning("Please upload a file or provide a link.")

elif task == "Classification":
    if method == "LLM (Large Language Model)":
        st.write("### Classification using LLM")
        if uploaded_file or file_link:
            if file_link:
                df = pd.read_csv(file_link)
            else:
                df = pd.read_csv(uploaded_file)

            st.write("### Dataset:")
            st.write(df.head())
            
            if "target" in df.columns:
                st.write("### LLM Classifier is not yet implemented.")
                st.warning("Please choose Data Mining Techniques for classification.")
            else:
                st.error("The dataset must contain a 'target' column for classification.")
        else:
            st.warning("Please upload a file or provide a link.")
    else:  # Data Mining Techniques (scikit-learn)
        st.write("### Classification using Data Mining")
        if uploaded_file or file_link:
            if file_link:
                df = pd.read_csv(file_link)
            else:
                df = pd.read_csv(uploaded_file)

            st.write("### Dataset:")
            st.write(df.head())

            if "target" in df.columns:
                st.write("### Training a Classifier...")
                X = df.drop("target", axis=1).select_dtypes(include=np.number)
                y = df["target"]
                model = RandomForestClassifier()
                model.fit(X, y)
                st.success("Model trained! Ready to classify new data.")
            else:
                st.error("The dataset must contain a 'target' column.")
        else:
            st.warning("Please upload a file or provide a link.")

elif task == "Regression":
    if method == "LLM (Large Language Model)":
        st.write("### Regression using LLM")
        st.warning("LLM-based regression is not implemented yet. Please choose Data Mining Techniques.")
    else:  # Data Mining Techniques (scikit-learn)
        st.write("### Regression using Data Mining")
        if uploaded_file or file_link:
            if file_link:
                df = pd.read_csv(file_link)
            else:
                df = pd.read_csv(uploaded_file)

            st.write("### Dataset:")
            st.write(df.head())

            if "target" in df.columns:
                st.write("### Training a Regressor...")
                X = df.drop("target", axis=1).select_dtypes(include=np.number)
                y = df["target"]
                model = LinearRegression()
                model.fit(X, y)
                st.success("Model trained! Ready to predict new data.")
            else:
                st.error("The dataset must contain a 'target' column.")
        else:
            st.warning("Please upload a file or provide a link.")

elif task == "Clustering":
    if method == "LLM (Large Language Model)":
        st.write("### Clustering using LLM")
        st.warning("LLM-based clustering is not implemented yet. Please choose Data Mining Techniques.")
    else:  # Data Mining Techniques (KMeans)
        st.write("### Clustering using Data Mining")
        if uploaded_file or file_link:
            if file_link:
                df = pd.read_csv(file_link)
            else:
                df = pd.read_csv(uploaded_file)

            st.write("### Dataset:")
            st.write(df.head())

            st.write("### Applying K-Means Clustering...")
            n_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)
            model = KMeans(n_clusters=n_clusters)
            df["cluster"] = model.fit_predict(df.select_dtypes(include=np.number))
            st.write("### Clustering Results:")
            st.write(df)
        else:
            st.warning("Please upload a file or provide a link.")

elif task == "Correlation Analysis":
    if method == "LLM (Large Language Model)":
        st.write("### Correlation Analysis using LLM")
        st.warning("LLM-based correlation analysis is not implemented yet. Please choose Data Mining Techniques.")
    else:  # Data Mining Techniques (Correlation Analysis)
        if (uploaded_file or file_link) and (uploaded_file_2 or file_link_2):
            if file_link:
                df1 = pd.read_csv(file_link)
            else:
                df1 = pd.read_csv(uploaded_file)

            if file_link_2:
                df2 = pd.read_csv(file_link_2)
            else:
                df2 = pd.read_csv(uploaded_file_2)

            st.write("### Correlation Analysis")
            st.write("Dataset 1:")
            st.write(df1.head())
            st.write("Dataset 2:")
            st.write(df2.head())

            st.write("### Correlation Results:")
            correlation = df1.corrwith(df2, axis=0)
            st.write(correlation)
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
            ## - [Social Network Ads Dataset](https://github.com/saeed-rhimi/K-NearestNeighbor/blob/main/Social_Network_Ads.csv)
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
