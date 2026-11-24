import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd

def Parser_URL_szo_old (url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')       
    years = soup.find_all('td', class_ = 'rg')
    titles = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'Середньорічна']
    content = dict()
    temp = [0]
    print('----------------------- Температура', url, '---------------------------------')
    for year in years:
        name = year.find('strong')
        timing = year.string
        try:
            timing = timing.replace(',', '.')
            timing = float(timing)
        except ValueError:
            pass
        if name != None:
            temp[0] = timing
        else:
            temp.append(timing)
        if len(temp) == 14:
            content[int(temp[0])] = temp[1:len(temp)]
            temp = [0]
    print('------------------------------------------------------------------------------')

    df = pd.DataFrame.from_dict(content, orient = 'index', columns = titles)
    df.index.name = 'Рік' 
    df['I'] = pd.to_numeric(df['I'], errors = 'coerce')
    df['IV'] = pd.to_numeric(df['IV'], errors = 'coerce')
    df['Середньорічна'] = pd.to_numeric(df['Середньорічна'], errors = 'coerce')
    df = df.fillna(np.nan)
    df['I'] = df['I'].fillna(round(df['I'].mean(), 1))
    df['IV'] = df['IV'].fillna(round(df['IV'].mean(), 1))
    df['Середньорічна'] = df['Середньорічна'].fillna(round(df['Середньорічна'].mean(), 1))
    df.to_csv(f'.\\output\\szo\\raw.csv')

    return

def Parser_URL_szo_new (url):
    i = 0
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')       
    years = soup.find('strong', string = 'Середні місячні температури повітря по м. Києву за багаторічний період')
    years = years.find_next('table', class_ = 'table-bar')
    years = years.find_all('td')
    years = years[14:-11]
    titles = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'Середньорічна']
    content = dict()
    temp = [0]
    print('----------------------- Температура', url, '---------------------------------')
    for year in years:
        timing = year.text
        try:
            timing = timing.replace(',', '.')
            timing = float(timing)
        except ValueError:
            pass

        if i > 13:
            i = 0
            content[int(temp[0])] = temp[1:len(temp)]

        if i == 0:
            temp = [timing]
        else:
            temp.append(timing)

        i +=1
    print('------------------------------------------------------------------------------')

    df = pd.DataFrame.from_dict(content, orient = 'index', columns = titles)
    df.index.name = 'Рік' 
    df['I'] = pd.to_numeric(df['I'], errors = 'coerce')
    df['IV'] = pd.to_numeric(df['IV'], errors = 'coerce')
    df['Середньорічна'] = pd.to_numeric(df['Середньорічна'], errors = 'coerce')
    df = df.fillna(np.nan)
    df['I'] = df['I'].fillna(round(df['I'].mean(), 1))
    df['IV'] = df['IV'].fillna(round(df['IV'].mean(), 1))
    df['Середньорічна'] = df['Середньорічна'].fillna(round(df['Середньорічна'].mean(), 1))
    df.to_csv(f'.\\output\\szo\\raw.csv')

    return