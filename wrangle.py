import pandas as pd
import requests
import os



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
        while year <= 2021:
            new_wines = requests.get(url+str(year)+'.json').json()
            wines.extend(new_wines)
            year += 1
        df = pd.DataFrame(wines)
        df.to_csv(filename)
        return df
    
    