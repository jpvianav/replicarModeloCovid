import json 
import pandas as pd

def getDataCol():
    dataCol = pd.read_json('https://www.datos.gov.co/resource/gt2j-8ykr.json')
    dataCol['dateRep'] = pd.to_datetime(dataCol['fecha_de_diagn_stico'], format='%d/%m/%Y')
    dataCol.sort_values(by = 'dateRep', inplace = True)
    return dataCol


def getDataWorld():
    try:
        data = pd.read_csv('https://opendata.ecdc.europa.eu/covid19/casedistribution/csv')    
        return data
    except:
        raise Exception('It was not possible to retrieve data from the european comission, please check internet connection of any change in the site address...')


if __name__ == '__main__':
    dataCol = getDataCol()
    dataWorld = getDataWorld()
    print('Hello')
