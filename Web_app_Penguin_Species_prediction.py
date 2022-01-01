import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Species Prediction App
This app predicts the **Palmer Penguin** species!

Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst""")

st.sidebar.header('User input features')
st.sidebar.markdown('You can also upload a csv file')

upload_file = st.sidebar.file_uploader('Upload a csv file', type= ['csv'])
if upload_file is not None:
    input_df = pd.read_csv(upload_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill Length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

penguins = pd.read_csv('penguins_cleaned.csv')
penguins = penguins.drop(['species'],axis=1)
penguins = pd.concat([input_df,penguins],axis=0)

encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(penguins[col],prefix=col)
    penguins = pd.concat([penguins,dummy],axis=1)
    del penguins[col]

penguins = penguins[0:1]

st.subheader('User Input Features')

if upload_file is not None:
    st.write(penguins)
else:
    st.write('Waiting for csv file to be uploaded. Hence, considering values given through interface.')
    st.write(penguins)

load_clf = pickle.load(open('penguins_clf.pkl','rb'))

prediction = load_clf.predict(penguins)
prediction_probability = load_clf.predict_proba(penguins)

st.write('Species Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
df = pd.DataFrame(columns=penguins_species)
prob_df = pd.DataFrame(prediction_probability)
prob_df.rename(columns = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'},inplace=True)
df = pd.concat([df,prob_df],axis=0)
st.write(prob_df)

print(prob_df.columns)