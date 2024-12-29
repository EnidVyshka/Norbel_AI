##Load libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly

#load model
model=joblib.load("Model/model.pkl")

# Define functions
def load_data(path):
    dataset = pd.read_csv(path)
    return dataset

# Load the dataset
data_path = "Dataset/train_data.csv"
load_df = load_data(data_path)
a = load_df.drop(columns=['locale', 'holiday', 'transferred', 'dcoilwtico', 'onpromotion'])
a.shitjet = a.shitjet.round()
test_df=pd.read_csv("Dataset/test.csv")

# Define section
data = st.container()

# Set up the data section that users will interact with
with data:
    data.title("Ne kete faqe mund te shikoni datasetet e juaja ne formen e nje tabele apo te nje grafiku te shperndare ne kohe ")
    st.write("Shfaq Dataset-in duke klikuar butonin e meposhtem")

    # Button to preview the dataset
    if st.button("Preview the dataset"):
        data.dataframe(a, use_container_width=True, hide_index=True)

    # Button to view the chart

    st.write("Shfaq grafikun")
    if st.button("View Chart"):

        # Set the "date" column as the index
        load_df = load_df.set_index('data')

        # Display the line chart with dates on the x-axis
        st.subheader("Grafiku i shitjeve ditore")
        st.line_chart(load_df["shitjet"])

