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

    context = """The lecturer and the convenor of the unit COS30081 Fundamentals of Natural Language Processing is Dr Joel Than Chia Ming. Student could contact Dr Joel Than Chia Ming by sending an email to jcmthan@swinburne.edu.my or through private message on Microsoft Teams. The unit COS30081 Fundamentals of Natural Language Processing is a twelve weeks long unit and it took up a semester time for student to complete the unit. This unit has two hours of lecture and two hours of tutorial a week from week one until week twelve. In total of 48 hours of contact hours expected for this unit. The first week of the unit start at 28 of February 2022. There are a total of ten pass tasks, one credit task and one distinction/high distinction task. The unit COS30081 Fundamentals of Natural Language Processing will cover the topic such as word tokenisation, TF-IDF vectors, Semantic Analysis, Deep Learning for natural language processing, basic and usage of convolutional neural network for Natural Language Processing, Recurrent Neural Network, Long Short Term Memory, Transformer, Performance Evaluation and Scaling up, and real world problem showcase. COS30081 Fundamentals of Natural Language Processing is a portfolio unit where there is no final exam for this unit. There are weekly lab tasks and two assignments to complete throughout the semester. Student are require to submit all the pass task in order to complete the unit. This is not a hard unit to pass if the student attend all the lecture and tutorial and finish weekly lab tasks. Some of the lab tasks can take some time to finish but it is not that hard to complete. COS30081 Fundamentals of Natural Language Processing deliver through two type of session which are online through Microsoft Team and face to face in Sarawak Campus. The unit is offered through blended learning. Student are require to complete COS20015 Fundamentals of Data Management and COS30019 Introduction to Artificial Intelligence before taking this unit as these are the prerequisite of this unit. Before taking COS30081 Fundamentals of Natural Language Processing here are the recommended reading material to go through which are Bengfort, B., Bilbro, R. and Ojeda, T. 2018. Applied text analysis with Python. O’Reilly, Boysolow II, T. 2018. Applied natural language processing with Python. Apress, and Cao, N. and Cui, W. 2016. Introduction to text visualization. Springer. It is recommended for student to have a basic knowledge of using Python with tensorflow before attempting this unit. COS30081 Fundamentals of Natural Language Processing is offered first time in Swinburne University of Technology Sarawak Campus. There are no improvement for the unit from the previous semester student feedback.  """

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