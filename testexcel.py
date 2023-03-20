import streamlit as st 
import pandas as pd 

st.title("excel update app')
df = pd.read_excel("QuestionsFinancedeMarche6.xlsx")
st.write(df)
