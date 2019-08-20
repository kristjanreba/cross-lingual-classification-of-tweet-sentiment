# Cross-lingual classification of tweet sentiment

In this repository you can find the code that was used to run experiments for my diploma thesis.


*BERT.ipynb* - contains the code for evaluation of multilingual BERT model. For the code to run we have to provide the data (read the file). It is recommended to run the experiments on you own machine using the GPU.

*separate.py* - additional script that separates text from labels into separate files.

*download_twitts.py* - script that downloads all the twitts by ID. We need to provide Twitter API keys and tweet IDs.

*clean_twitts.py* - script that cleans the provided tweets. It also merges the tweets with the labels

In the *LASER* folder there is a python script called *embed_all.py* that creates embeddings with the help of LASER library for each specified language.
The folder also contains file *classification_LASER.py* where experiments with LASER+ML are defined.

In the *data* folder there is a folder *twitter_sentiment* containing twitt IDs with coresponding hand labels. We can use those IDs to download the twitts using *download_twitts.py*.
