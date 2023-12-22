import nltk
nltk.download('punkt')
nltk.download('stopwords')

import json
import streamlit as st
import pickle
import string
import requests
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from streamlit_lottie import st_lottie
ps=PorterStemmer()


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_sequrity = load_lottieurl("https://lottie.host/89a5ca58-b9b1-4741-8bce-af10035bccf3/wgJtW757GH.json")
# lottie_messege=load_lottieurl("https://lottie.host/df085eaf-f9cc-4ba8-8255-4f07c866f1dc/xOo0YN4q5v.json")

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")
# st_lottie(
#         lottie_messege,
#         speed=1,
#         reverse=False,
#         loop=True,
#         quality="low", # medium ; high
#         # renderer="svg", # canvas
#         height=100,
#         width=100,
#         key=None,
#     )










input_sms=st.text_area("enter the message")
if st.button('predict'):
    # ''' 
    # 1) preprocess
    # 2) vectorize
    # 3) predict
    # 4) display
    # '''

    # 1)
    transformed_sms=transform_text(input_sms)
    # 2) vectorize
    vector_input=tfidf.transform([transformed_sms])
    # 3) predict
    result=model.predict(vector_input)[0]
    # 4) result display
    if(result==1):
        st.header("Spam")
    else:
        st.header("Not Spam")



with st.container():
    st.markdown("""
    <style>
        .st-l {
            display: inline-block; /* Display icons side by side */
            margin-right: 10px; /* Adjust spacing as needed */
        }
    </style>
    """, unsafe_allow_html=True)

    st_lottie(
        lottie_sequrity,
        speed=1,
        reverse=False,
        loop=True,
        quality="low", # medium ; high
        # renderer="svg", # canvas
        height=100,
        width=100,
        key=None,
    )
    
