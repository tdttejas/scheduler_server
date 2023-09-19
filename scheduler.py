from elasticsearch import Elasticsearch
from datetime import datetime
import logging
import os

es = Elasticsearch(["http://localhost:9200"])

# date = datetime.now().strftime("%Y"+".%m"+".%d-*")
date = "2023.09.18-*"
index_pattern = ".ds-filebeat-8.10.0-"+date

count=0
# timestamp="2023-09-12T14:05:44.124Z"
timestamp = datetime.now().isoformat()

with open("last_timestamp.txt","r") as file:
    timestamp = file.read()

query = {
    "query": {
        "range": {
            "@timestamp":{
                "gte": timestamp
            }
        }
    },
    "size": 10000
}

try:
    response = es.search(index=index_pattern, body=query)

    # print(response["hits"]["hits"])
    curr_timestamp = datetime.now().isoformat()
    with open("last_timestamp.txt","w") as file:
        file.write(curr_timestamp)
        
    dataset_dir = "dataset"  
    os.makedirs(dataset_dir, exist_ok=True)
    os.chdir(dataset_dir)
    logging.basicConfig(
        filename='HDFS.log',
        format='',
        level=logging.INFO
    )
    if len(response["hits"]["hits"])>0 : 
        for hit in response["hits"]["hits"]:
            logging.info(hit["_source"]["message"])
            print(hit["_source"]['message'])

except Exception as e:
    print(f"An error occurred: {e}")

