import pandas as pd
import numpy as np
import llm_helper as llmh
import plotly.express as px
from sklearn.metrics import silhouette_score
import nltk
import PyPDF2
import os
import pdfplumber
from sklearn.preprocessing import StandardScaler
import document_uploader as docup
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


def classification(target_column_name=None, support_column_names=None, file_link="./temp/current-file.csv", dataframe=None, user_query=None):
    if target_column_name == None or target_column_name == "":
        target_column_name = "target"
    if dataframe is None:
        df = pd.read_csv(file_link)
    else:
        df = dataframe
    if support_column_names == None:
        X = df.drop(columns=[target_column_name])
    else:
        X = df[support_column_names]
    y = df[target_column_name]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.to_list())
    X_test = scaler.transform(X_test.to_list())

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)



    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
    }

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    labels = sorted(y.unique())

    # Create a DataFrame for the confusion matrix
    conf_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    # Plot confusion matrix with Plotly
    fig = px.imshow(conf_df,
                    text_auto=True,
                    labels={'x': "Predicted", 'y': "Actual", "color": "Count"},
                    title="Confusion Matrix")
    fig.update_layout(xaxis_title="Predicted Labels", yaxis_title="Actual Labels")
    
    if user_query:
        metrics["Prediction to user query"] = model.predict([llmh.get_text_embedding([user_query])])
    return fig, metrics


def clustering(n_clusters=3, file_link = "./temp/current-file.csv", dataframe=None):
    
    if dataframe is None:
        df = pd.read_csv(file_link)
    else:
        df = dataframe
    
    df_numeric = df.select_dtypes(include=["number"])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_numeric)
    
    silhouette_avg = silhouette_score(df_numeric, df['Cluster'])
    metrics = {
        "Silhouette Score": silhouette_avg,
    }
    print("Silhouette Score:", silhouette_avg)
    
    # Plot clusters with Plotly
    if df_numeric.shape[1] >= 2:  # Only plot if there are at least 2 numeric columns
        fig = px.scatter(df, x=df_numeric.columns[0], y=df_numeric.columns[1], 
                         color='Cluster', title="KMeans Clustering", color_continuous_scale='Greys',
                         labels={'Cluster': 'Cluster'})
        fig.update_traces(marker=dict(size=8))
    else:
        fig = None  # No plot if less than 2 numeric features
    
    return fig, metrics


def regression(target_column_name="target", file_link = "./temp/current-file.csv"):
    
    # Load dataset
    df = pd.read_csv(file_link)
    
    # Separate features and target variable
    X = df.drop(columns=[target_column_name])
    y = df[target_column_name]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        "Mean Squared Error": mse,
        "R-squared": r2,
    }
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                     title="Actual vs Predicted Values")
    fig.update_traces(marker=dict(size=8))
    
    return fig, metrics


def metrics_to_str(metrics):
    result = ""
    for key, value in metrics.items():
        result += f"#### {key}: {value}\n"
    return result


def correlation_analysis(file_link = "./temp/current-file.csv"):
    df = pd.read_csv(file_link).select_dtypes(include=["number"])
    correlation_matrix = df.corr()
    
    corr_long = correlation_matrix.stack().reset_index(name="correlation")
    corr_long.columns = ['Feature 1', 'Feature 2', 'Correlation']
    
    fig = px.density_heatmap(corr_long, x='Feature 1', y='Feature 2', z='Correlation', 
                             color_continuous_scale='Greys', 
                             color_continuous_midpoint=0, 
                             title="Correlation Heatmap")
    
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Features",
        xaxis={'categoryorder': 'category descending'},
        yaxis={'categoryorder': 'category ascending'}
    )

    return fig


def summarize_text(document_name=None, input=None, num_sentences=7):
    nltk.download('punkt_tab')
    if document_name:
        text = docup.text_from_document(document_name)
    elif input == None or input == "":
        return
    else:
        text = input
    sentences = sent_tokenize(text)    
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    sentence_scores = cosine_sim.sum(axis=1)
    
    ranked_sentences = [i for i in range(len(sentence_scores))]
    ranked_sentences.sort(key=lambda x: sentence_scores[x], reverse=True)
    
    summary = [sentences[i] for i in ranked_sentences[:num_sentences]]
    
    result = """Top important sentences: 
    """
    iterator = 0
    for sentence in summary:
        iterator = iterator + 1
        result += f"""
        
{iterator}: {sentence}"""
    
    return result



