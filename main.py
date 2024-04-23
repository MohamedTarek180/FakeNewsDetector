import pandas as pd
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

MNB = MultinomialNB()
K10 = KNeighborsClassifier(10)
LR = LogisticRegression()
cv = CountVectorizer(max_features=1000)
counter = 0

st.set_page_config(layout='wide', page_title="Did It Happen?", page_icon='❓', )
st.markdown("<h1 style='text-align: center; font-size: 42px; color: white;'>Did it Happen❓</h1>",
            unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def getData():
    dfReal = pd.read_csv("True.csv")
    dfFake = pd.read_csv("Fake.csv")
    return dfReal, dfFake


# st.markdown("<h1 style='text-align: center; font-size: 30px; color: white;'>An exercise in NLP\nBy Ahmed Ashraf</h1>",
#             unsafe_allow_html=True)

textDecided = st.text_input("Please input Text you want to check")


@st.cache(allow_output_mutation=True)
def readData():
    real, fake = getData()

    real['label'] = 0
    fake['label'] = 1

    df = pd.concat([fake, real])
    df.drop(['title', 'subject', 'date'], axis=1, inplace=True)

    text = df['text']
    label = df['label']
    return text, label


text, label = readData()
words = cv.fit_transform(text).toarray()
columns = np.array(cv.get_feature_names_out())
newDF = pd.DataFrame(columns=columns, data=words)

MNB.fit(newDF, label)
K10.fit(newDF, label)
LR.fit(newDF, label)

newwords = cv.transform([textDecided]).toarray()
val1 = MNB.predict(newwords)
val2 = K10.predict(newwords)
val3 = LR.predict(newwords)

if val1[0] == 1:
    st.markdown(f"Naive Bayes: Fake")
    counter += 1
else:
    st.markdown(f"Naive Bayes: True")

if val2[0] == 1:
    st.markdown(f"KNN: Fake")
    counter += 1
else:
    st.markdown(f"KNN: True")

if val3[0] == 1:
    st.markdown(f"Logistic regression: Fake")
    counter += 1
else:
    st.markdown(f"logistic regression: True")

if counter >= 2:
    st.markdown("<h1 style='text-align: center; color: red;'>Fake</h1>", unsafe_allow_html=True)

else:
    st.markdown("<h1 style='text-align: center; color: green;'>True</h1>", unsafe_allow_html=True)
