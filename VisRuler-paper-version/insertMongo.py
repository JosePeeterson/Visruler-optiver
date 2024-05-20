#!/usr/bin/env python
import sys
import pandas as pd
import pymongo
import json
import os


# def import_content(filepath):
#     mng_client = pymongo.MongoClient('localhost', 27017)
#     mng_db = mng_client['mydb']
#     #collection_name = 'StanceCTest' 
#     collection_name = 'HappUnlab' 
#     db_cm = mng_db[collection_name]
#     cdir = os.path.dirname(__file__)
#     file_res = os.path.join(cdir, filepath)

#     data = pd.read_csv(file_res)
#     data_json = json.loads(data.to_json(orient='records'))
#     db_cm.remove()
#     db_cm.insert(data_json)

def import_content(filepath):
    mng_client = pymongo.MongoClient('localhost', 27017)
    mng_db = mng_client['mydb']
    #collection_name = 'StanceCTest' 
    collection_name = 'optiver_100' 
    db_cm = mng_db[collection_name]
    cdir = os.path.dirname(__file__)
    file_res = os.path.join(cdir, filepath)

    data = pd.read_csv(file_res)
    data_json = json.loads(data.to_json(orient='records'))
    
    # Delete existing documents
    db_cm.delete_many({})  # or db_cm.delete_one({}) if you want to delete only one document
    
    # Insert new documents
    db_cm.insert_many(data_json)  # or db_cm.insert_one(data_json) if you have only one document


if __name__ == "__main__":
  filepath = 'C:\Finance_projects\VisRuler-paper-version\VisRuler-paper-version\data\optiver_100.csv'
  import_content(filepath)