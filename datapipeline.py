import json
import pandas as pd

<<<<<<< HEAD
def DataMapper():
    with open('./data/satelite mapper/satelitecode_mapper.json', 'r') as f:
        map_data = json.load(f)
    return map_data

def DataPipeline(data):
    data_ = data.dropna()
    map_data_satelite_code = DataMapper()
=======
def DataMapper(maploc):
    with open(maploc, 'r') as f:
        map_data = json.load(f)
    return map_data

def DataPipeline(data, maploc):
    data_ = data.dropna()
    map_data_satelite_code = DataMapper(maploc)
>>>>>>> localdev
    data_['Satelite_Code'] = data_['Satelite_Code'].map(map_data_satelite_code)
    return data_