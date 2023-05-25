import pandas as pd
import os
import datetime
import re
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
from streamlit_image_select import image_select




#for ML 
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

import streamlit as st 
# from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
# import spacy
from spacy import displacy
# nlp = spacy.load('en')
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

#for email 

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(to_email, subject, body):
    # Replace 'your_email@gmail.com' and 'your_email_password' with your email and password
    gmail_user = 'ju.michoux@gmail.com'
    gmail_password = 'xxx'

    message = MIMEMultipart()
    message['From'] = gmail_user
    message['To'] = to_email
    message['Subject'] = subject

    # Add body to email
    message.attach(MIMEText(body, 'plain'))

    # Send email using Gmail SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(gmail_user, gmail_password)
    text = message.as_string()
    server.sendmail(gmail_user, to_email, text)
    server.quit()
	
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

p_filedirectory = 'QuestionsFinancedeMarche10.xlsx' #CHANGEMENT
dataframe_all = pd.ExcelFile(p_filedirectory)
dataframe_allsheets = dataframe_all.sheet_names
dataframe_allsheets.remove('Commentaires  ') #delete sheet Commentaires


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

    if current_row_possibilities == "INPUT": #give the possibility to answer by writting (see elif)
        message = st.text_area("Enter your text : ", "")
        if st.button("Summarize"): 
          real_answer = current_row['Answer'].iloc[0]#useless to check 
          real_justification = current_row['Justification'].iloc[0]#useless to justification
#           sentences = [
#               message,
#               real_answer
#               ] #for the model
#           sentence_embeddings = model.encode(sentences) #modèle intermédiaire
#           similitude = util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1]) #result of the model 
#           if similitude > 0.5 : 
#             st.success("Exactement ! Quelques compléments : \n " + str(real_justification))
#           else : 
#             st.error("Faux ! puisque : " + str(real_justification))

    
    elif current_row_possibilities == "Image": #give the possibility to answer by writting
        st.write("gg")
        message = st.write("Choose your answer from the following possibilities", "")
        img2 = image_select("Label", ["images/Question_6_Options/CallOK.JPG", "images/Question_6_Options/Put.JPG"])
#         if img == "images/Question_6_Options/Put.JPG" : 
#             st.write("gg")

    else : 
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


def primarychoice():
    # Get user email input
    user_email = st.text_input("Enter your email")

    # Check if email is valid
    if not re.match(r"[^@]+@[^@]+\.[^@]+", user_email):
        st.warning("Please enter a valid email address")
    else:
        # Send email
        def save_email(email):
            with open("mails.txt", "a") as f:
                f.write(email + "\n")
        st.success("Email is valid !")
        ################TEST 
        from PIL import Image
#         test_path = "1000_F_42570032_tNXjCF0k7hUSojxE6kyuFKPKC6NjJAZ2.jpg"
#         image = Image.open(test_path)
#         st.image(image)  # Colonne contenant le texte de légende de l'image dans le fichier Excel

        img1 = image_select("Label", ["images/Question_6_Options/CallOK.JPG", "images/Question_6_Options/Put.JPG"])
        if img1 == "images/Question_6_Options/Put.JPG" : 
            st.write("gg")

        ################TEST

        
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
            /* Sidebar */
            [data-testid="stSidebar"]  {
                background-color: #F9F9F9;
                border-right: 1px solid #D2D2D2;
            }

            /* Main Content Area */
            [data-testid="stAppMain"] {
                background-color: #FFFFFF;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            }

            /* Page Title */
            h1 {
                font-size: 48px;
                color: #242424;
                font-family: Arial, sans-serif;
                background-color: #60b1f7;
                padding: 10px; /* Ajouter cette ligne pour donner de l'espace autour du titre */
            }

            /* Section Headings */
            h2 {
                font-size: 36px;
                color: #333333;
                font-family: Arial, sans-serif;
                margin-bottom: 10px;
            }

            /* Subsection Headings */
            h3 {
                font-size: 24px;
                color: #333333;
                font-family: Arial, sans-serif;
                margin-bottom: 10px;
            }

            /* Paragraphs */
            p {
                font-size: 18px;
                color: #4A4A4A;
                font-family: Arial, sans-serif;
                line-height: 1.5;
                margin-bottom: 10px;
            }

            /* Buttons */
            .stButton {
                background-color: #45A160;
                color: #FFFFFF;
                border-radius: 5px;
                border: none;
                font-size: 18px;
                padding: 10px 20px;
                margin-top: 20px;
                margin-bottom: 20px;
                font-family: Arial, sans-serif;
                cursor: pointer;
            }

            .stButton:hover {
                background-color: #4EB36B;
            }

            /* Text Input Fields */
            .stTextInput {
                border-radius: 5px;
                border: 1px solid #D2D2D2;
                font-size: 18px;
                padding: 10px;
                margin-top: 10px;
                margin-bottom: 10px;
                font-family: Arial, sans-serif;
                color: #4A4A4A;
            }

            /* Checkbox Input Fields */
            .stCheckbox {
                margin-top: 10px;
                margin-bottom: 10px;
            }

            /* Select Input Fields */
            .stSelectbox {
                border-radius: 5px;
                border: 1px solid #D2D2D2;
                font-size: 18px;
                padding: 10px;
                margin-top: 10px;
                margin-bottom: 10px;
                font-family: Arial, sans-serif;
                color: #4A4A4A;
                background-color: #FFFFFF;
            }

            /* Select Input Field Options */
            .stSelectbox option {
                font-size: 18px;
                font-family: Arial, sans-serif;
                color: #4A4A4A;
            }
            [data-testid="stSidebar"]  {background-color: #e5e5f7;
            opacity: 0.9;
            background-image:primaryColor="#6eb52f",backgroundColor="#f0f0f5",secondaryBackgroundColor="#e0e0ef",textColor="#262730",font="sans serif";)
            #background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #e5e5f7 20px ), repeating-linear-gradient( #5245f755, #5245f7 );}

            [data-testid="stAppViewContainer"]  {background-color: #e5e5f7;
            opacity: 0.9;
            background-image:  radial-gradient(#f7a645 1px, transparent 1px), radial-gradient(#f7a645 1px, #e5e5f7 1px);
            background-size: 40px 40px;
            background-position: 0 0,20px 20px;
            }
        </style>
        
        """
        st.markdown(page, unsafe_allow_html=True)

        return(df_current)

if __name__ == '__main__':
    primarychoice()
