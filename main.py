import streamlit as st
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import warnings
# import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from nltk.corpus import stopwords
# import string
# from wordcloud import WordCloud

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
# from sklearn import metrics

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import docx2txt
from nltk.tokenize import WhitespaceTokenizer

import plotly.graph_objects as go
import plotly.express as px

# import chart_studio.plotly as py

import re

import PyPDF2
import os

warnings.filterwarnings('ignore')


header = st.container()



with header: 
    st.title('Welcome to the Awsome Resume Analyser')

df = pd.read_csv('data/Dataset/UpdatedResumeDataSet.csv', encoding='utf-8')

#preprocessing
resumeDataSet = df.copy()
resumeDataSet['cleaned_resume'] = ''

#resumetext cleaning function
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

#call the function
resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

#Encoding labels into different values
var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])


#splitting of dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)


clf = KNeighborsClassifier(n_neighbors=15)
clf = clf.fit(X_train, y_train)
yp = clf.predict(X_test)

class JobPredictor:
    def __init__(self) -> None:
        self.le = le
        self.word_vectorizer = word_vectorizer
        self.clf = clf

    def predict(self, resume):
        feature = self.word_vectorizer.transform([resume])
        predicted = self.clf.predict(feature)
        resume_position = self.le.inverse_transform(predicted)[0]
        return resume_position

    def predict_proba(self, resume):
        feature = self.word_vectorizer.transform([resume])
        predicted_prob = self.clf.predict_proba(feature)
        return predicted_prob[0]
    
user_input = st.text_area("Enter Job Description you applied", )
job_description = user_input


resume_position = JobPredictor().predict(job_description)

text_tokenizer= WhitespaceTokenizer()
remove_characters= str.maketrans("", "", "±§!@#$%^&*()-_=+[]}{;'\:,./<>?|")
cv = CountVectorizer()

def pdfToText(path):
    pdfreader = PyPDF2.PdfFileReader(path)
    no_of_pages = pdfreader.numPages
    with open('final_txt.txt', 'w') as f:
        for i in range(0, no_of_pages):
            pagObj = pdfreader.getPage(i)
            f.write(pagObj.extractText())
    with open('final_txt.txt', 'r') as f:
        text = f.read()
    if os.path.exists("final_txt.txt"):
        os.remove("final_txt.txt")
        return text

fileType = st.selectbox(
            'Select your file type',
            ['Document', 'PDF']
        )

text = None

if fileType == 'PDF':
    st.write('Selected file type is ', fileType)
    pdf_file = st.file_uploader(label="Upload PDF File")
    if pdf_file is not None: 
        text = pdfToText(pdf_file)
        st.success('PDF Uploaded Susscessfully')

if fileType == 'Document': 
    st.write('Selected file type is ', fileType)
    docx_file = st.file_uploader(label="Upload Document File")
    if docx_file is not None:
        text = docx2txt.process(docx_file)
        st.success('Document Uploaded Susscessfully')

if text is None: 
    st.error('File is not upladed')

if text is not None: 
    #takes the texts in a list
    text_docx= [text, job_description]
    #creating the list of words from the word document
    words_docx_list = text_tokenizer.tokenize(text)
    #removing speacial charcters from the tokenized words 
    words_docx_list=[s.translate(remove_characters) for s in words_docx_list]
    #giving vectors to the words
    count_docx = cv.fit_transform(text_docx)
    #using the alogorithm, finding the match between the resume/cv and job description
    similarity_score_docx = cosine_similarity(count_docx)
    match_percentage_docx= round((similarity_score_docx[0][1]*100),2)
    st.write('Match percentage with the Job description:', match_percentage_docx)
    fig1 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = match_percentage_docx,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Match with JD"}))
    st.plotly_chart(fig1)
    job_predictor = JobPredictor()
    resume_position = job_predictor.predict(text)

    chart_data = pd.DataFrame({
        "position": [cl for cl in job_predictor.le.classes_],
        "match": job_predictor.predict_proba(text)
    })

    fig = px.bar(chart_data, x="position", y="match",
                    title=f'Resume matched to: {resume_position}')
    st.plotly_chart(fig)





