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
    # set the filename
    filename = 'wines.csv'
    # if the file is already locally saved
    if os.path.isfile(filename):
        # return it
        return pd.read_csv(filename).iloc[:,1:]
    # if it is not locally saved
    else:
        # get the first page of data at this url
        url = 'https://top100.winespectator.com/wp-content/themes/top100-theme/src/data/1988.json'
        # save it as a dictionary
        wines = requests.get(url).json()
        # set the base url for the follow-on pages of data
        url = 'https://top100.winespectator.com/wp-content/themes/top100-theme/src/data/'
        # setup variables for the loop
        wines = []
        year = 1988
        # run a loop to grab the rest of the pages of data
        while year <= 2019:
            # grab the data from the page as a dictionary
            new_wines = requests.get(url+str(year)+'.json').json()
            # extend the wines list with the dictionary
            wines.extend(new_wines)
            # add one to year to get the next page
            year += 1
        # make the list of dictionaries a dataframe
        df = pd.DataFrame(wines)
        # save it locally as a csv
        df.to_csv(filename)
        # return it
        return df
    
def prepare(df):
    '''
    This function takes in the winespectator.com dataset as a dataframe and returns it after
    it does the following:
    - removes nulls
    - creates dummies for country and color
    - corrects pricing abnormalities and handles outliars
    - renames columns
    - lemmitizes the tasters note
    '''
    # drop columns with a lot of nulls or irrelevant data
    df = df.drop(columns=['winery_full_nonentities', 'alternate_bottle_size', 'taster_initials', 'id'])
    # set 'no vintage'/blends to have average vintage year
    filler = round(df[df.vintage != 'NV'].vintage.astype(int).mean())
    df.vintage = df.vintage.replace('NV', filler).astype(int)
    # correct pricing to match rest of pricing
    df.price = df.price.replace('35 / 375ml', '70').replace('49 / 500ml', '73.5').astype(float)
    # fill null wine colors to red based on exploration findings
    df.color = df.color.fillna('red').str.lower()
    
    # set up for loop
    new = []
    # take only the year from the issue date
    for date in df.issue_date:
        new.append(date[8:])
    # reset it to the feature
    df.issue_date = new
    # make it an int
    df.issue_date = df.issue_date.astype(int)
    
    # rename columns
    df.columns = ['winery', 'wine', 'vintage', 'note', 'color', 'country', 'region', 'score', 'price', 'issue_year', 'top100_year', 'top100_rank']
    # remove price outliars
    df = df[(df.price < 350)&(df.price > 5)]
    # feature engineer aged as the difference between when the grapes were harvested and when the bottle was available for purchase
    df['aged'] = abs(df.issue_year - df.vintage)
    
    # make dummies for the country
    dummies = pd.get_dummies(df.country, dtype=int)
    # only keep the dummies with over 100 bottles in the dataset
    dummies = dummies[['France','Italy','California','Spain','Australia','Washington']]
    # add it to the end of the main dataframe
    df = pd.concat([df, dummies], axis=1)
    
    # simplify the colors to be red, white, and other
    df.color = df.color.map({'red':'red', 'white':'white', 'dessert':'other', 'sparkling':'other', 'blush':'other'})
    # make dummies for the color
    dummies = pd.get_dummies(df.color, dtype=int)
    # add it to the end of the main dataframe
    df = pd.concat([df, dummies], axis=1)
    
    # drop extra columns that dummies were created for
    df = df.drop(columns=['country', 'region', 'color', 'other'])


    # create and add to stopword list
    stopword_list = stopwords.words('english')
    stopword_list.append('wine')
    stopword_list.append('red')
    # create lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    # set up for the loop
    results = []
    # loop through all the tasters notes for each wine bottle
    for note in df.note:
        # make the note lowercase
        note = note.lower()
        # normalize it, encode it, decode it
        note = unicodedata.normalize('NFKD', note).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        # save only letters
        note = re.sub(r"[^a-z'\s]", '', note)
        # use the lemmatizer
        lemmas = [wnl.lemmatize(word) for word in note.split()]
        # bring it all back together
        note_lemmatized = ' '.join(lemmas)
        # look at each of the words
        words = note.split()
        # only save the words not in the stopword list
        filtered = [w for w in words if w not in stopword_list]
        # bring it back together
        note_without = ' '.join(filtered)
        # add it back to the list to be rejoined to the main dataframe
        results.append(note_without)
    # put the cleaned version back in the dataframe
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
    '''
    This function takes in the winespectator.com dataset as a dataframe and
    returns a dataframe with key words from the taster's note made into 
    dummies based on frequency.
    '''
    # make it
    t = TfidfVectorizer()
    # fit it
    X_nlp = t.fit_transform(df.note)
    y = df.price
    # save the results in a dataframe
    results = pd.DataFrame(X_nlp.todense(), columns=t.get_feature_names_out())
    # make all the 0s into nulls
    results = results.replace(0.0, np.nan)
    # drop columns beyond the treshold
    results = results.dropna(axis='columns', thresh=319)
    # return all the nulls back to 0s
    results = results.fillna(0)
    # make the index match
    results.index = df.index
    # put them together
    df = pd.concat([df, results], axis=1)
    
    return df