import streamlit as st

from multiapp import MultiApp
# import your pages here
from pages import model_1, model_2, model_3

# Create an instance of the app
app = MultiApp()

# Title of the main page
st.title("Education Chatbot")

# Add all your applications (pages) here
app.add_app("Bert-Cased-SQuAD", model_1.app)
app.add_app("Distilbert-Cased-SQuAD", model_2.app)
app.add_app("DistilRoBERTa-Base-SQuAD", model_3.app)
# The main app
app.run()
