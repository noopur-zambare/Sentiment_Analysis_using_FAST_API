import pandas as pd
import os
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



class SemtimentDataset:
    def __init__(self, pth_to_csv="airline_sentiment_analysis.csv"):
        self.data = pd.read_csv(pth_to_csv)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data['airline_sentiment'])

        print('Pre-processing the data...')
        self.data['text'] = self.preprocess_text(self.data.text)
        self.data['airline_sentiment'] = self.label_encoder.transform(self.data['airline_sentiment'])

        self.train_split, self.test_split = train_test_split(self.data, test_size=0.2, random_state=42, stratify=self.data.airline_sentiment)
    
    def preprocess_text(self, data):
        lemmatizer = WordNetLemmatizer()
        stopword = set(stopwords.words('english'))

        clean_data = []
        for sentence in tqdm(data):
            cleantext = BeautifulSoup(sentence, "lxml").text 
            cleantext = re.sub(r'[^\w\s]','',cleantext)
            cleantext = [token for token in cleantext.lower().split() if token not in stopword]
            clean_text = ' '.join([lemmatizer.lemmatize(token) for token in cleantext])
            clean_data.append(clean_text.strip())

        return clean_data
    
    def get_train_data(self):
        return self.train_split['text'], self.train_split['airline_sentiment']
    
    def get_test_data(self):
        return self.test_split['text'], self.test_split['airline_sentiment']


class SentimentModel:
    def __init__(self, base_clf=None, base_clf_params={}, vectorizer=None):
        self.base_clf = base_clf(**base_clf_params) if base_clf is not None else LogisticRegression(C=1,solver='saga')
        self.vectorizer = vectorizer or CountVectorizer()
    
    def fit(self, x_train, y_train):
        print('Fitting the model...')
        x_transformed = self.vectorizer.fit_transform(x_train)
        self.base_clf.fit(x_transformed, y_train)

        preds = self.base_clf.predict(x_transformed)
        acc = accuracy_score(y_train, preds)
        print(classification_report(y_train, preds))

        return acc
    
    def test(self, x_test, y_test):
        print('Testing the model...')
        x_transformed = self.vectorizer.transform(x_test)
        preds = self.base_clf.predict(x_transformed)
        acc = accuracy_score(y_test, preds)

        print('Testing Accuracy: {}'.format(acc))
        print(classification_report(y_test, preds))

    
    def save(self, pth_to_save='./'):
        print('Saving the model...')
        dump(self.base_clf, os.path.join(pth_to_save, 'model.joblib'))
        dump(self.vectorizer, os.path.join(pth_to_save, 'vectorizer.joblib'))



if __name__ == "__main__":
    dataset = SemtimentDataset()
    x_train, y_train = dataset.get_train_data()
    x_test, y_test = dataset.get_test_data()

    model = SentimentModel(vectorizer=CountVectorizer())
    
    train_acc = model.fit(x_train, y_train)
    print('Training Accuracy: {}'.format(train_acc))

    model.test(x_test, y_test)

    model.save()

    print('Done!')