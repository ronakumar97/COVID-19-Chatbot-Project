import streamlit as st
import spacy
import json
from chatbot_backend import preprocessing, tokenize_data, create_categorical_target, train_model
import random
import tensorflow as tf
import numpy as np
from streamlit_chat import message

def get_response(input_text):
    sent_seq = []
    doc = nlp(repr(input_text))

    for token in doc:
        if token.text in tokenizer.word_index:
            sent_seq.append(tokenizer.word_index[token.text])
        else:
            sent_seq.append(tokenizer.word_index['<unk>'])

    sent_seq = tf.expand_dims(sent_seq, 0)
    pred = model(sent_seq)

    pred_class = np.argmax(pred.numpy(), axis=1)
    return random.choice(intent_doc[trg_index_word[pred_class[0]]]), trg_index_word[pred_class[0]]

@st.cache(allow_output_mutation=True)
def train_chatbot(tokenizer, input_tensor, target_tensor):
    return train_model(tokenizer, input_tensor, target_tensor)

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')

    with open('intents.json') as f:
        intents = json.load(f)

    inputs, targets = [], []
    classes = []
    intent_doc = {}

    for intent in intents['intents']:
        if intent['intent'] not in classes:
            classes.append(intent['intent'])
        if intent['intent'] not in intent_doc:
            intent_doc[intent['intent']] = []

        for text in intent['text']:
            inputs.append(preprocessing(text))
            targets.append(intent['intent'])

        for response in intent['responses']:
            intent_doc[intent['intent']].append(response)

    tokenizer, input_tensor = tokenize_data(inputs)
    target_tensor, trg_index_word = create_categorical_target(targets)

    model = train_chatbot(tokenizer, input_tensor, target_tensor)

    if "history" not in st.session_state:
        st.session_state.history = [
            {"message": 'Welcome. I am CoronaBot. How can I help ?', "is_user": False}
        ]

    st.title('COVID-19 Chatbot')

    input_text = st.text_input(label="Ask me anything...")

    if(input_text):
        st.session_state.history.append({"message": input_text, "is_user": True})
        result, type = get_response(input_text)
        print('INTENT TYPE -> {}'.format(type))
        print('RESPONSE -> {}'.format(result))

        if(result):
            st.session_state.history.append({"message": result, "is_user": False})

    count = 0

    for chat in st.session_state.history:
        message(**chat, key=str(count))
        count += 1






