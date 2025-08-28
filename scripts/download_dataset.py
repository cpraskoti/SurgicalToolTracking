
# -*- coding: utf-8 -*-
"""
Download CholecTrack20 Dataset
author: Chinedu I. Nwoye
"""     
import requests
import synapseutils
import synapseclient

# Load configuration from environment variables
from dotenv import load_dotenv
import os

load_dotenv()

def main():

    # 1. Login with Synapse credentials
    print("Authenticating user ...")
    syn = synapseclient.login(email=email, authToken=authToken)

    # 2. Request entity access
    print("Authenticating access key permission to download dataset ...")
    API_URL = "https://synapse-response.onrender.com/validate_access"
    USER_ID = syn.getUserProfile()['ownerId']
    response = requests.post(API_URL, json={"access_key": accesskey, "synapse_id": USER_ID})
    if response.status_code == 200:
        entity_id = response.json()['entity_id']
    else:
        print("❌ Failed to request access:", response.text)
        exit(1)

    # 3. Download dataset to your local folder
    print("Downloading dataset...")
    _ = synapseutils.syncFromSynapse(syn, entity=entity_id, path=local_folder)
    print("success!")


if __name__ == "__main__":


    email = os.getenv('SYNAPSE_EMAIL')  # Your email address in the Synapse account
    authToken = os.getenv('SYNAPSE_AUTH_TOKEN')  # Your authToken from Synapse account
    accesskey = os.getenv('CHOLECTRACK20_DATASET_ACCESS_KEY')  # Your dataset access key ID
    local_folder = os.getenv('DATASET_LOCAL_PATH', 'data/cholectrack20')  # Local folder for downloaded data

    main()
