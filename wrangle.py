import numpy as np
import pandas as pd
import requests
import os
from sklearn.model_selection import train_test_split
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer




def acquire ():
    '''
    This function checks if the wine data is saved locally. 
    If it is not local, this function reads the win data from 
    winespectator.com and return it in a DataFrame.'''
    filename = 'wines.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename).iloc[:,1:]
    else:
        url = 'https://top100.winespectator.com/wp-content/themes/top100-theme/src/data/1988.json'
        wines = requests.get(url).json()
        url = 'https://top100.winespectator.com/wp-content/themes/top100-theme/src/data/'
        wines = []
        year = 1988
        while year <= 2019:
            new_wines = requests.get(url+str(year)+'.json').json()
            wines.extend(new_wines)
            year += 1
        df = pd.DataFrame(wines)
        df.to_csv(filename)
        return df
    
def prepare(df):
    df = df.drop(columns=['winery_full_nonentities', 'alternate_bottle_size', 'taster_initials', 'id'])
    filler = round(df[df.vintage != 'NV'].vintage.astype(int).mean())
    df.vintage = df.vintage.replace('NV', filler).astype(int)
    df.price = df.price.replace('35 / 375ml', '70').replace('49 / 500ml', '73.5').astype(float)
    df.color = df.color.fillna('red').str.lower()
    # df.issue_date = pd.to_datetime(df.issue_date, format='%b %d, %Y')
    new = []
    for date in df.issue_date:
        new.append(date[8:])
    df.issue_date = new
    df.issue_date = df.issue_date.astype(int)
    df.columns = ['winery', 'wine', 'vintage', 'note', 'color', 'country', 'region', 'score', 'price', 'issue_year', 'top100_year', 'top100_rank']
    df = df[(df.price < 350)&(df.price > 5)]
    df['aged'] = abs(df.issue_year - df.vintage)
    
    dummies = pd.get_dummies(df.country, dtype=int)
    dummies = dummies[['France','Italy','California','Spain','Australia','Washington']]
    df = pd.concat([df, dummies], axis=1)
    df.color = df.color.map({'red':'red', 'white':'white', 'dessert':'other', 'sparkling':'other', 'blush':'other'})
    dummies = pd.get_dummies(df.color, dtype=int)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=['country', 'region', 'color', 'other'])

    
    stopword_list = stopwords.words('english')
    stopword_list.append('wine')
    stopword_list.append('red')


    wnl = nltk.stem.WordNetLemmatizer()

    results = []
    for note in df.note:
        note = note.lower()
        note = unicodedata.normalize('NFKD', note).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        note = re.sub(r"[^a-z'\s]", '', note)
        lemmas = [wnl.lemmatize(word) for word in note.split()]
        note_lemmatized = ' '.join(lemmas)
        words = note.split()
        filtered = [w for w in words if w not in stopword_list]
        note_without = ' '.join(filtered)
        results.append(note_without)
    df.note = results
    
    
    return df
    
    
def split_data(df):
    '''
    Takes in a dataframe and target (as a string). Returns train, validate, and test subset 
    dataframes with the .2/.8 and .25/.75 splits to create a final .2/.2/.6 split between datasets
    '''
    # split the data into train and test. 
    train, test = train_test_split(df, test_size = .2, random_state=123)
    
    # split the train data into train and validate
    train, validate = train_test_split(train, test_size = .25, random_state=123)
    
    return train, validate, test

def add_nlp(df):
    t = TfidfVectorizer()
    X_nlp = t.fit_transform(df.note)
    y = df.price
    results = pd.DataFrame(X_nlp.todense(), columns=t.get_feature_names_out())
    results = results.replace(0.0, np.nan)
    results = results.dropna(axis='columns', thresh=319)
    results = results.fillna(0)
    results.index = df.index
    df = pd.concat([df, results], axis=1)
    
    return df