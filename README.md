# Cross-lingual classification of tweet sentiment

In this repository you can find the code that was used to run experiments for my diploma thesis.

### Install:
- Facebooks LASER library

```
conda install numpy pandas scikit-learn
```

In the *data* folder there is a folder *twitter_sentiment* containing twitt IDs with coresponding hand labels. We can use those IDs to download the twitts using *download_twitts.py*.

*download_twitts.py* - script that downloads all the twitts by ID. We need to provide Twitter API keys and tweet IDs.
(If you need the whole dataset write to authors of this article https://www.clarin.si/repository/xmlui/handle/11356/1054.
We are only alowed to distribute it on individual basis because of Twitter policy.)

*clean_twitts.py* - script that cleans the provided tweets. It also merges the tweets with the labels

*separate.py* - additional script that separates text from labels into separate files.

*mBERT.ipynb* - contains the code for evaluation of multilingual BERT model. For the code to run we have to provide the data (read the file). It is recommended to run the experiments on your own machine using a GPU.

*CSE_BERT.ipynb* - contains the code for evaluation of CroSloEngual BERT model. For the code to run we have to provide the data (read the file). It is recommended to run the experiments on your own machine using a GPU.


In the *LASER* folder there is a python script called *embed_all.py* that creates embeddings with the help of LASER library for each specified language.
The folder also contains file *classification_LASER.py* where experiments with LASER+ML are defined.


Code might need some adjustment depending on what experiments you want to run, what data you want to use and what platform you are using.
