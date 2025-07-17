import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *


df = pd.read_csv('../datasets/college_student_placement_dataset.csv')

st.title('College Student Placement Prediction')

df.drop('College_ID', axis=1, inplace=True)
df['Internship_Experience'] = df['Internship_Experience'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Placement'] = df['Placement'].apply(lambda x: 1 if x == 'Yes' else 0)


# st.write(df)


x = df.drop('Placement', axis=1)
y = df['Placement']

model = LogisticRegression(max_iter=1000)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

keys = df.columns.tolist()

values = {}

for key in keys:
    if key != 'Placement':
        values[key] = st.number_input(f'Enter {key}', value=0)

if st.button('Predict Placement'):
    input_data = pd.DataFrame([values])
    prediction = model.predict(input_data)
    st.write('Placement Prediction:', 'Yes' if prediction[0] == 1 else 'No')
    
