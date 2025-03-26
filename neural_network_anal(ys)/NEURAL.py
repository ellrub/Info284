#############
# CONVERZZION FUNXXION

import pandas;

def dataframeConverzzion(_dataset):
    _mergedReviews = [];
    for i in _dataset.to_numpy():
        _mergedReviews.append([ i[2], i[0] + " " + i[1] ]);
    #return pandas.DataFrame(_mergedReviews);
    return _mergedReviews;



#############
# LOADING DATASET

import pandas;

dataset = pandas.read_csv("../../Hotel_Reviews_small.csv", usecols = ["Negative_Review", "Positive_Review", "Reviewer_Score"]);
dataset = dataframeConverzzion(dataset);



#############
# DATASET PREPARATION FUNXXIONS

import re;
import nltk;

nonWords = nltk.corpus.stopwords.words("english");
nltk.download("stopwords");
nltk.download("punkt_tab");
nltk.download("wordnet");
lemma = nltk.WordNetLemmatizer();

def stringClean(_string):
    _string = _string.lower();
    _string = re.sub(r"\d+", "", _string);
    _string = re.sub(r"[^\w\s]", "", _string);
    return _string;

def lemmaBalls(_string):
    _text = nltk.word_tokenize(_string);
    _text_without_trash = [];
    for _word in _text:
        if not (_word in nonWords):
            _text_without_trash.append(lemma.lemmatize(_word));
    return _text_without_trash;

def datasetPreparation(_dataset):
    for _review in _dataset:
        _review[1] = lemmaBalls(stringClean(_review[1]));
    return _dataset;



#############
# PREPARING DATASET

datasetPrepared = datasetPreparation(dataset);



#############
# ENCRYPXION FUNXXION

# so the word fragment usually in shaders and filters and stuff refers to 
# one point on the texture or surface or whatever you are rendering to
# basicly like a pixel, but a little more abstract
# im gonna be running a convolutional filter on the dataset
# and i want each word to be one fragment for the filter
# it kinda sucks to have spaces and words of varying lengths because then the filter
# has to figure out dynamically where one fragment ends and begins,
# meaning it probly has to read ahead until the next space or something and that sucks
# so im turning words into fragments of fixed size
# so that the filter knows that a fragment is always exactly say 4 characters
# so then it doesnt need to figure it out on its own, it doesnt even need spaces between words
# so then im kinda turning the words into "pixels" for the filter

# put all the words into a set
# count how many entries there are in the set
# calculate how many digits in base 64 would be needed to uniquely identify words
# make a relation from b64 keys to words
# make a relation from words to b64 keys
# replace words in dataset with b64 keys
# remove spaces

import math;

def datasetEncrypxion(_dataset):
    _wordSet = {};
    for _review in _dataset:
        for _word in _review:
            _wordSet.add(_word);
    
    _requiredDigits = math.floor(math.log(len(_wordSet), 64)) + 1;
    _wordList = [];

    for _word in _wordSet:
        _wordList.add(_word);

    _wordsRelateKeys = {};
    _keysRelateWords = {};

    for _i in range(len(_word)):
        _key = 
        _wordsRelateKeys[_]
        
