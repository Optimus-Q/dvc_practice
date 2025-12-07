import json
import pytest
import pandas as pd
from datapipeline import DataPipeline

def test_datamapper():
    mapper_loc = "./data/satelite mapper/satelitecode_mapper.json"
    with open(mapper_loc, 'r') as f:
        map_data = json.load(f)
    map_type = type(map_data)
    req_type = {"G10": 10}
    req_type = type(req_type)
    assert req_type == map_type

def test_datapipeline():
    mapper_loc = "./data/satelite mapper/satelitecode_mapper.json"
    dataset_loc = "./data/train/GNSS_raw_train.csv"
    df = pd.read_csv(dataset_loc)
    df_updated = DataPipeline(df, mapper_loc)
    check_nan = int(df_updated['Satelite_Code'].isna().sum(axis=0))
    check_map = [1 if isinstance(x, str) else 0 for x in df_updated['Satelite_Code']]
    check_map = sum(check_map)
    assert check_nan == 0
    assert check_map == 0

