import streamlit as st 
import pandas as pd 

st.title("excel update app")
df = pd.read_excel("QuestionsFinancedeMarche6.xlsx")
st.write(df)

dataframe_all = pd.ExcelFile("QuestionsFinancedeMarche6.xlsx")
dataframe_allsheets = dataframe_all.sheet_names
st.write(dataframe_allsheets)

