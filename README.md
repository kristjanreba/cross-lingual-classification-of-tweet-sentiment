# Cross-lingual classification of tweet sentiment

In this repository you can find the code that was used to run experiments presented in Cross-lingual Transfer of Twitter Sentiment Models [Todo: add link to paper].

### Install:
- Facebooks LASER library
- use mbert.yaml and csebert.yaml to create Anaconda environments for each of the bert models


## Contents of this repo:
In the *data* folder there is a folder *twitter_sentiment* containing twitt IDs with coresponding hand labels. We can use those IDs to download the twitts using *download_twitts.py*.

- *download_twitts.py* - script that downloads the twitts by ID (many twitts are no longer available online so this script will not download the whole dataset). 
You need to provide Twitter API keys and tweet IDs.
(If you need the whole dataset, write to authors of this paper https://www.clarin.si/repository/xmlui/handle/11356/1054.
We are only alowed to distribute it on individual basis because of Twitter policy.)

- *clean_twitts.py* - script that cleans the provided tweets. It also merges the tweets with the labels

- *separate.py* - additional script that separates text from labels into separate files.

- *mbert.py* - contains the code for evaluation of multilingual BERT model. For the code to run we have to provide the data (read the file). It is recommended to run the experiments on your own machine using a GPU. (Run this file in conda environemnt created with mbert.yaml)

- *cse_bert.py* - contains the code for evaluation of CroSloEngual BERT model. For the code to run we have to provide the data (read the file). It is recommended to run the experiments on your own machine using a GPU. (Run this file in conda environemnt created with csebert.yaml)

- In the *LASER* folder there is a python script called *embed_all.py* that creates embeddings with the help of LASER library for each specified language.
The folder also contains file *classification_LASER.py* where experiments with LASER+ML are defined.


Code might need some adjustment depending on what experiments you want to run, what data you want to use and what platform you are using.
