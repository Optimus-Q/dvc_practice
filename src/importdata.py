import boto3
import argparse
from pathlib import Path
import logging
import yaml
import os

def loadconfig(path: str = "params.yaml") -> dict:
    with open(path, "r") as yf:
        return yaml.safe_load(yf)
    
def s3downloadfile(client, bucket, filekey, targetpath):
    targetpath.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, filekey, str(targetpath))

def main(client, params):
    s3_bucket = params["s3bucketname"]
    s3_trainkey = params["s3trainkey"]
    s3_testkey = params["s3testkey"]
    s3_datamapper = params["s3datamapper"]
    s3_traintargetpath = Path(s3_trainkey)
    s3_testtargetpath = Path(s3_testkey)
    s3_datamappertargetpath = Path(s3_datamapper)
    s3downloadfile(client, s3_bucket, s3_trainkey, s3_traintargetpath)
    s3downloadfile(client, s3_bucket, s3_testkey, s3_testtargetpath)
    s3downloadfile(client, s3_bucket, s3_datamapper, s3_datamappertargetpath)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    s3_client = boto3.client("s3")
    loadparams = loadconfig(args.params)
    main(client=s3_client, params=loadparams)

    