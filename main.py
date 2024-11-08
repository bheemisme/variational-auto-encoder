import streamlit as st
from compression import compression
from decompression import decompression
import matplotlib.pyplot as plt
import pandas as pd

def get_graph():
    # Read the CSV file
    results_df = pd.read_csv(f"results/results_train_11.csv")

    # Create the plot
    fig = plt.figure(figsize=(6,4))
    plt.plot(results_df["epoch_no"], results_df["train_loss"], label="Train Loss")
    plt.plot(results_df["epoch_no"], results_df["test_loss"], label="Test Loss")

    # Add labels and title
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Test Loss")

    # Add legend
    plt.legend()
    return fig

def app():
    st.title("Neural Lossy Compression")
    df = pd.read_csv('./results/results_train_20.csv')
    fig = get_graph()
    st.pyplot(fig)
    
    
    compression()
    decompression()


app()