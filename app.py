import streamlit as st

from multiapp import MultiApp
# import your pages here
from pages import model_1, model_2, model_3, model_4, model_5, model_6
# Create an instance of the app
app = MultiApp()

# Title of the main page
st.title("Education Chatbot")

# Add all your applications (pages) here
app.add_app("Bert-Cased-SQuAD-V1", model_1.app)
app.add_app("Distilbert-Cased-SQuAD-V1", model_2.app)
app.add_app("DistilRoBERTa-Base-SQuAD-V1", model_3.app)
app.add_app("Bert-Cased-SQuAD-V2", model_4.app)
app.add_app("Distilbert-Cased-SQuAD-V2", model_5.app)
app.add_app("DistilRoBERTa-Base-SQuAD-V3", model_6.app)
# The main app
app.run()
