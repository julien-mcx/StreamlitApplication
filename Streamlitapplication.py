import pandas as pd
import os
import datetime
from datetime import datetime, timedelta
import warnings
import io
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
# from streamlit_image_comparison import image_comparison
import plotly.express as px  # interactive charts
from PIL import Image
import json

#for ML 
import streamlit as st 
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import spacy
from spacy import displacy
nlp = spacy.load('en')

# import spacy 
# from gensim.summarization import summarize
# from textblob import textBlob
# from gensim.summarization import summarize

#for vizualisation
import requests
from streamlit_lottie import st_lottie
import base64
def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )

p_filedirectory = 'QuestionsFinancedeMarche7.xlsx' #CHANGEMENT
dataframe_all = pd.ExcelFile(p_filedirectory)
dataframe_allsheets = dataframe_all.sheet_names


def load_lottiefile(filepath :str):
    with open(filepath, "r") as f : 
        return json.load(f)


def treatment(p_dataframe):
    df_current = p_dataframe
    #display description 
    current_description = df_current['Description'][0]
    st.write(current_description) #à retravailler mise en forme ? 

    #choice questions
    all_questions = df_current['Question Number '].unique()
    question_selection = st.selectbox("Choose your question : ",all_questions) 

    df_current_without = df_current.drop('Description', axis=1)#keep all columns without 'Description'

    #keep important row 
    current_row = df_current_without.loc[df_current_without['Question Number ']==question_selection]

    #display question
    current_row_question = current_row['Questions'].iloc[0]
    # st.markdown(f'<p style="background-color:#24243e;color:#d0000;font-size:24px;border-radius:2%;">{current_row_question}</p>', unsafe_allow_html=True)
    st.write(f"**{current_row_question}**")

    #possibilities 
    current_row_possibilities = current_row['Possibility'].iloc[0]
    current_split_row_possibilities = list(current_row_possibilities.split(","))

    if current_row_possibilities != "INPUT": #give the possibility to answer by writting (see elif)

        #current answer
        answer = st.selectbox("Choose your answer :", current_split_row_possibilities)
        if st.button("Send"):
            #check answer
            real_answer = current_row['Answer'].iloc[0]#useless to check 
            real_justification = current_row['Justification'].iloc[0]#useless to justification        
            
            if answer == real_answer :
                st.success("Exactement ! Quelques compléments : \n " + real_justification)

            else : 
                st.error("Faux ! puisque : " + real_justification)
    
    elif current_row_possibilities == "INPUT": #give the possibility to answer by writting
        message = st.text_area("Enter your text : ", "")
        if st.button("Summarize"):
            st.write("gg")
            
#             st_lottie(load_lottiefile("\\\\ad-its.credit-agricole.fr\\dfs\\HOMEDIRS\\AMUNDI\\michoux\\Desktop\\Personnel\\Projets Python\\Questions d’entretiens en Finance de Marché\\versiongithub\\lottiefiles\\hello.json"), speed = 1, reverse=False, loop = True, quality  = "low")
            # tokens = pegasus_tokenizer(message, truncation = True, padding = "longest", return_tensors = "pt")            


def primarychoice():

    #choix de la sheet associée à la sélection
    st.sidebar.info("This app is maintained by Michoux Julien. " "You can contact me at [ju.michoux@gmail.com].")
    st.sidebar.title("Let's choose your asset type")
    l_assettypechoice = st.sidebar.radio("Choose your asset type : ", dataframe_allsheets)
    df_current = pd.read_excel(p_filedirectory, sheet_name=l_assettypechoice)


    #traitement
    st.title(l_assettypechoice)
    treatment(df_current)

    #vizualization 
    page = """
    <style>
    [data-testid="stSidebar"]  {background-color: #e5e5f7;
opacity: 0.9;
background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #e5e5f7 20px ), repeating-linear-gradient( #5245f755, #5245f7 );}

    [data-testid="stAppViewContainer"]  {background-color: #e5e5f7;
opacity: 0.9;
background-image:  radial-gradient(#f7a645 1px, transparent 1px), radial-gradient(#f7a645 1px, #e5e5f7 1px);
background-size: 40px 40px;
background-position: 0 0,20px 20px;
    }
    <style>
    """
    st.markdown(page, unsafe_allow_html=True)


    return(df_current)
    

if __name__ == '__main__':
    primarychoice()
