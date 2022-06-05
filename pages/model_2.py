import streamlit as st
from streamlit_chat import message as st_message
import transformers
from transformers import pipeline


def app():
    st.subheader('Distilbert-Cased-SQuAD')

    @st.cache(allow_output_mutation=True)
    def load_model():

        model = pipeline("question-answering",
                         model="victorlee071200/distilbert-base-cased-finetuned-squad")

        return model

    context = """Welcome to COS30081 Fundamentals of Natural Language Processing. My name is Dr. Joel Than and I will be the convener, lecturer, and tutor of this unit. Fundamentals of Natural Language Processing is a 12-week unit with the following pre-requisites COS20015 Fundamentals of Data Management and COS30019 Introduction to Artificial Intelligence. This unit introduces students to the essential natural language processing (NLP) tasks and techniques. It equips students with skills to carry out basic text data pre-processing, feature extraction, building and evaluating a text classifier, and visualizing NLP results. This unit also exposes students to an advanced NLP technique. For the Week 1 lecture, students will have a soft introduction to Natural Language Processing (NLP). NLP is a growing area of research and has a strong presence in different industries. NLP plays a pivotal role in introducing artificial intelligence to different applications. By the end of week 1, students should be able to Identify key applications of Natural Langauge Processing, Define Regular Expressions (Regex), and Describe the NLP pipeline. For the Week 2 lecture, students will embark on the first major component of the NLP pipeline which is Word Tokens. Imagine that a document is a large bill. Word tokenization separates that bill into tokens that can be inputted into the rest of the NLP pipeline just like a slot machine. With word tokens, we can identify the frequency of occurrence of a particular word in a document. By the end of week 2, the student should be able to identify the key concepts of Word Tokens, identify the challenges of tokenization, and describe the fundamentals of a "word". For the Week 3 lecture, students will discover the counts can be misleading and may not be that informative. Students will also be introduced to the term Term Frequency - Inverse Document Frequency (TF-IDF). By the end of week 3, the student should be able to Identify the key concepts of TF-IDF, Define Zipf's Law, and Execute TF-IDF after word tokenization. For the Week 4 lecture, students will try to process the TF-IDF of documents for sentiment analysis, students will explore the usage of libraries such as sklearn and GENSIM along with converting TF-IDFs into Topic Vectors that can be coupled with a classifier. By the end of week 4, student should be able to Identify the meaning of semantic analysis, Apply several approaches for semantic analysis, and Explain key differences between PCA and LDiA. For the Week 5 lecture, it will be covered the first model that pushed attention to the usage of Word2Vec which features a 2-layer neural network. Besides that, students will also be looking at GloVE which takes a different approach and looks at the co-occurrence matrix. Students will also be reviewing FastText which is an extension of Word2Vec that goes down to the character level on Week 5. By the end of week 5, student should be able to Identify key concepts of using Word2Vec, GloVE, FastText, Apply Word2Vec, GloVE, FastText, and Explain the key advantages of the three models. For the Week 6 lecture, students will be utilizing the word vectors with Convolutional Neural Networks (CNN)s. CNNs were conventionally used and made popular by the imaging community as it has gained prominence in creating a feature map that is more powerful than handcrafted features. The key idea behind CNNs is the convolution layers that have a convolution operation with a filter. By the end of week 6, student should be able to Identify the key concepts of CNNs for NLP, Design a basic CNN for NLP classification, and Describe the different layers of CNNs that are used for NLP. For Week 7 lecture, students are venturing to the land of networks that greatly benefit from Sequential data. Previously, students were hovering in the realm of CNN applications which were initially designed for images. Now we will see the improvement RNNs bring by bringing the recurrent information from previous words. Through the lecture, the student will understand the weakness of RNN that is overcome with LSTMs. By the end of week 7 students should be able to identify the key concepts of RNNs and LSTM, design a basic RNN and LSTM network for NLP classification, and describe the difference between RNN and LSTM. For the Week 8 lecture students are going into the realm of sequence-to-sequence model and transformers. These models have shown great potential in machine translation with Transformers now leading the way in deep learning across various fields. Attention with Seq2Seq models has also great promise. These models typically have an encoder and a decoder the outputs from the encoder are used as inputs for the decoder. By the end of week 8, you should be able to Identify the key concepts of Encoders and Decoders, Understand the key differences between Seq2Seq and Transformers, Describe the fundamentals of Transformers. For the Week 9 lecture, students are exploring the idea of pretraining for transformers/NLP. LSTM models do not benefit from this. As seen for Computer Vision applications, pre-trained models can help increase performance and give more context to words. We will also explore fine-tuning these pre-trained models for specific tasks such as sentiment analysis. There are three approaches for pre-training which included the pre-train the encoder stack, pre-train a decoder stack, and the pre-train the entire encoder-decoder stack. By the end of week 9, the student should be able to Identify the concepts of BERT and GPT, Describe the difference between encoder and decoder pretraining, and define the difference between pretraining and fine-tuning. For the Week 10 lecture, student will be exploring further pre-trained networks and their performance. We look specifically at the cost of training them. Besides that, students will explore the popular dataset for Question Answering Tasks. Thus, we will also explore a fine-tuned model approach for Question Answering. By the end of week 10, students should be able to Identify the top-performing transformers and models today, Describe the complex relationship between performance and complexity/parameters, and Describe details of Question Answering and SQuAD dataset. For the week 1 tutorial, the student will do some simple tasks and refresh the student’s memory in using the Python language and as well as a soft introduction using Regex. For the week 2 tutorial, the student will do some simple tasks of tokenization. For the week 3 tutorial, the student will start with the basics of running TF-IDF from scratch followed by using sklearn packages for an interesting movie recommendation. For the Week 4 tutorial students will try to process the TF-IDF of documents for sentiment analysis. Students will explore the usage of libraries such as sklearn and GENSIM. For the Week 5 tutorial, students will mainly look at the application of three pre-trained word vector models like word2vec, GloVE, and fastText. For the Week 6 tutorial, students will look at the basics of using Convolutional Neural Networks for Natural Language Processing along with showing the student the fundamental steps in preparing for datasets, initializing the model, train and testing the model. Besides, students will also explore saving and reloading the model with its weights. In short, week 6 will cover Deep learning libraries for CNN, Preprocessing steps for Deep Learning, DL Model Initialisation, and Compilation, Training, and Testing. For Credit Task, the task involves skills that students have obtained for the first half of the semester. For the Week 7 tutorial, students will look at the basics of using recurrent neural networks and Long Short-Term Memory (LSTM) Network. Students will learn the fundamental steps in utilizing the Keras layers that are used for RNN and LSTM to achieve classification or next character prediction. For the week 8 tutorial, the student will look at the applications of Sequence-to-Sequence Models and Transformers for Machine Translation which also cover the application of Seq2Seq for Machine Translation and the Application of Transformer for Machine Translation. For the week 9 tutorial student will look at another application of finetuning pre-trained transformers specifically for question answering which also cover Accessing pre-trained BERT models from Hugging Face, Understanding Question and Answering, and Applying a pre-trained BERT model for Questions and Answering. For the Distinction/ High Distinction Task, the task required the student to use deep learning techniques to create a question-and-answer chatbot for the scenario below. Use at least two different deep networks such as BERT-base (Transformer) to create a question-and-answer feature of a chatbot. For Pass Task 1, the due date will be Week 3 Sunday, 11:59 PM. For Pass Task 2, the due date will be Week 4 Sunday, 11:59 PM. The due date for Pass Task 3 will be Week 5 Sunday, 11:59 PM. For Pass Task 4, the due date will be Week 6 Sunday, 11:59 PM. For Pass Task 5 and Credit Task, the due date will be Tuition Week Sunday, 11:59 PM. For Pass Task 6, the due date will be Week 8 Sunday, 11:59 PM. For Pass Task 7, the due date will be Week 9 Sunday, 11:59 PM. For Pass Task 8, the due date will be Week 10 Sunday, 11:59 PM. For Pass Task 9, the due date will be Week 11 Sunday, 11:59 PM. For the Distinction Task (D), the due date will be Week 12 Sunday, 11:59 PM. For the High Distinction (HD) Task, the due date will be Week 12 Sunday, 11:59 PM."""

    bot_message_count = 1
    user_message_count = 2

    qa = load_model()

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:

        st.text('Chatbot Configuration')

        details_options = ['Yes', 'No']

        show_all_results = st.radio('Show result details', details_options)

    def generate_answer():

        user_message = st.session_state.input_text

        if user_message:

            answers = qa(question=user_message, context=context)

            message_bot = answers['answer']

            if show_all_results == 'Yes':
                with st.sidebar:

                    st.write("Answer: " + str(answers['answer']))
                    st.write("Score: " + str(answers['score']))

        st.session_state.history.append(
            {"message": user_message, "is_user": True, "key": user_message_count})
        st.session_state.history.append(
            {"message": message_bot, "is_user": False, "key": bot_message_count})

    st.text_input("Ask me anything about COS30081 Fundamentals of Natural Language Processing",
                  key="input_text", on_change=generate_answer)

    for chat in st.session_state.history:

        st_message(**chat)  # unpacking

        user_message_count += 1
        bot_message_count += 1
