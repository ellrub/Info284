#############
# CONVERZZION FUNXXION

import pandas;

def dataframeConverzzion(_dataset):
    _mergedReviews = [];
    for i in _dataset.to_numpy():
        _mergedReviews.append({ "score": i[2], "review": i[0] + " " + i[1] });
    #return pandas.DataFrame(_mergedReviews);
    return _mergedReviews;



#############
# LOADING DATASET

import pandas;

dataset = pandas.read_csv("../../Hotel_Reviews_small.csv", usecols = ["Negative_Review", "Positive_Review", "Reviewer_Score"]);
dataset = dataframeConverzzion(dataset);



#############
# DATASET EXTRAXXION

def entryExtraxxion( _entry, _col ):
    _entryExtracc = [];
    for _wordExtracc in _entry[_col]:
        _entryExtracc.append(_wordExtracc);
    return _entryExtracc;

def datasetExtraxxion( _dataset, _col):
    _datasetExtracc = [];
    for _entry in _dataset:
        _entryExtracc = entryExtraxxion(_entry, _col);
        _datasetExtracc.append(_entryExtracc);
    return _datasetExtracc;



#############
# DATASET PREPARATION

import re;
import nltk;

nonWords = nltk.corpus.stopwords.words("english");
nltk.download("stopwords");
nltk.download("punkt_tab");
nltk.download("wordnet");
lemma = nltk.WordNetLemmatizer();
stemma = nltk.PorterStemmer();

def stringClean( _string ):
    _string = _string.lower();
    _string = re.sub(r"\d+", "", _string);
    _string = re.sub(r"[^\w\s]", "", _string);
    return _string;

def lemmaBalls( _string ):
    _text = nltk.word_tokenize(_string);
    _textLemma = [];
    for _word in _text:
        _wordLemma = lemma.lemmatize(_word);
        if not (_wordLemma in nonWords):
            _textLemma.append(_wordLemma);
    return _textLemma;

def stemmaBalls( _string ):
    _text = nltk.word_tokenize(_string);
    _textStemma = [];
    for _word in _text:
        _wordStemma = stemma.stem(_word);
        if not (_wordStemma in nonWords):
            _textStemma.append(_wordStemma);
    return _textStemma;

def datasetPreparation( _dataset, _mode =  "lemma" ):
    _datasetPrepare = [];
    if( _mode == "lemma" ):
        for _review in _dataset:
            _reviewPrepare = { "score": _review["score"], "review": lemmaBalls(stringClean(_review["review"])) };
            _datasetPrepare.append(_reviewPrepare);
    elif( _mode == "stemma" ):
        for _review in _dataset:
            _reviewPrepare = { "score": _review["score"], "review": stemmaBalls(stringClean(_review["review"])) };
            _datasetPrepare.append(_reviewPrepare);
    else:
        raise ValueError("provided mode does not exist");

    return _datasetPrepare;



#############
# PREPARING DATASET

datasetPrepared = datasetPreparation(dataset, _mode = "stemma");



#############
# DATASET VECTORIZATION

def entryVecc( _entry, _model ):
    _entryVecc = [];
    for _word in _entry:
        if( _word in _model.wv ):
            _wordVecc = _model.wv[_word];
            _entryVecc.append(_wordVecc);
    return _entryVecc;

def datasetVectorization( _dataset, _model ):
    _datasetVecc = [];
    for _review in _dataset:
        _reviewVecc = { "score": _review["score"], "review": entryVecc(_review["review"], _model) };
        _datasetVecc.append(_reviewVecc);
    return _datasetVecc;




#############
# VECTORIZING DATASET

from gensim.models import Word2Vec;

fragmentChannels = 100;

datasetExtracc = datasetExtraxxion(datasetPrepared, "review");

# Create CBOW model
bowModel = Word2Vec(datasetExtracc, min_count = 1, vector_size = fragmentChannels, window = 5);
datasetVeccBow = datasetVectorization(datasetPrepared, bowModel);

# Create Skip Gram model
skipModel = Word2Vec(datasetExtracc, min_count = 1, vector_size = fragmentChannels, window = 5, sg = 1);
datasetVeccSkip = datasetVectorization(datasetPrepared, skipModel);



#############
# SPLITTING DATASET

from sklearn.model_selection import train_test_split

dataframeVeccBow = pandas.DataFrame(datasetVeccBow);
dataframeVeccSkip = pandas.DataFrame(datasetVeccSkip);

veccBowtrainX, veccBowTexts, veccBowtrainY, veccBowTestY = train_test_split(dataframeVeccBow["review"].values, dataframeVeccBow["score"].values, test_size = 0.20, random_state = 1);
veccSkiptrainX, veccSkipTestX, veccSkiptrainY, veccSkipTestY = train_test_split(dataframeVeccSkip["review"].values, dataframeVeccSkip["score"].values, test_size = 0.20, random_state = 1);

print(veccSkiptrainX[0]);
print(veccSkipTestX[0]);
print(veccSkiptrainY[0]);
print(veccSkipTestY[0]);



#############
# DEFINING NETWORK

import tensorflow;
from tensorflow.keras import layers, datasets, models, callbacks;

with tensorflow.device("/GPU:0"):
    model = models.Sequential();
    model.add(layers.Input( (None, fragmentChannels)) );
    model.add(layers.Conv1D( 32, 4, activation = "relu", strides = 1 ));
    model.add(layers.BatchNormalization());
    model.add(layers.Dense( 96, activation = 'relu' ));
    model.add(layers.Dropout( 0.2 ));
    model.add(layers.Dense( 192, activation = 'relu' ));
    model.add(layers.Dropout( 0.3 ));
    model.add(layers.Dense( 128, activation = 'relu' ));
    model.add(layers.Dropout( 0.2 ));
    model.add(layers.Dense( 10, activation = 'relu' ));

    model.compile( optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] );

    early_stopping = callbacks.EarlyStopping( monitor = 'val_loss', patience = 5, restore_best_weights = True );

    history = model.fit(
        veccSkiptrainX,
        veccSkiptrainY,
        batch_size = 64,
        epochs = 50,
        validation_data = (veccSkipTestX, veccSkipTestY),
        callbacks = [early_stopping]
    );


# use flatten layer if youre gonna do several sentences in parrallel



#############
# VISUALIZATION /// EVALUATION 

from matplotlib import pyplot;

pyplot.plot(history.history['accuracy'], label='accuracy');
pyplot.plot(history.history['val_accuracy'], label = 'val_accuracy');
pyplot.xlabel('Epoch');
pyplot.ylabel('Accuracy');
pyplot.ylim([0.5, 1]);
pyplot.legend(loc='lower right');

loss, accuracy = model.evaluate(veccSkipTestX,  veccSkipTestY, verbose=2);
