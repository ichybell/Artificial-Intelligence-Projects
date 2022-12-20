# Sentiment Analysis

The project was aimed at being able to train a machine learning model to enable it to be able to classify a sentence as either a positive or negative. 
The code in the Jupyter file, `Sentiment.ipynb` was the original code implementation using only two datasets where one was used a training set and the other as a test set. The performance was low despite using multiple classification techniques. 
Methods of improving the performance are documented at the end of the file.

The code in `sentiment.py` was meant to illustrate how it would be done if we combined all the datasets into one and split the resultant set into two with a training set and a test set. The test set would then be used to test the performance and it provided a better performance because of the increase in the size of the dataset.

The rest of the README was extracted from the original location of the dataset: **[Sentiment Labelled Sentences](https://archive.ics.uci.edu/ml/datasets/sentiment+labelled+sentences)**

This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015

It contains sentences labelled with positive or negative sentiment, extracted from reviews of products, movies, and restaurants

=======
Format:
=======
sentence \t score \n


=======
Details:
=======
Score is either 1 (for positive) or 0 (for negative)	
The sentences come from three different websites/fields:

imdb.com
amazon.com
yelp.com

For each website, there exist 500 positive and 500 negative sentences. Those were selected randomly for larger datasets of reviews. 
We attempted to select sentences that have a clearly positive or negative connotaton, the goal was for no neutral sentences to be selected.



For the full datasets look:

imdb: Maas et. al., 2011 'Learning word vectors for sentiment analysis'
amazon: McAuley et. al., 2013 'Hidden factors and hidden topics: Understanding rating dimensions with review text'
yelp: Yelp dataset challenge http://www.yelp.com/dataset_challenge