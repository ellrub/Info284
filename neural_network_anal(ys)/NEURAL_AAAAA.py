#############
# DATAFRAME CONVERZZION

import pandas;

def dataframeConverzzion( _dataset ):
    _mergedReviews = [];
    for i in _dataset.to_numpy():
        _mergedReviews.append({ "score": i[2], "review": i[0] + " " + i[1] });
    #return pandas.DataFrame(_mergedReviews);
    return _mergedReviews;



#############
# DATASET EXTRAXXION

from typing import Iterable

def entryExtraxxion( _entry, _col ):
    _entryExtracc = [];
    if( isinstance(_entry[_col], Iterable) ):
        for _word in _entry[_col]:
            _entryExtracc.append(_word);
        return _entryExtracc;
    else:
        return _entry[_col];

def datasetExtraxxion( _dataset, _col ):
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
    _entryLongest = 0;
    if( _mode == "lemma" ):
        for _review in _dataset:
            _reviewPrepare = { "score": _review["score"], "review": lemmaBalls(stringClean(_review["review"])) };
            _datasetPrepare.append(_reviewPrepare);
            if( len(_reviewPrepare["review"]) > _entryLongest ):
                _entryLongest = len(_reviewPrepare["review"]);
    elif( _mode == "stemma" ):
        for _review in _dataset:
            _reviewPrepare = { "score": _review["score"], "review": stemmaBalls(stringClean(_review["review"])) };
            _datasetPrepare.append(_reviewPrepare);
            if( len(_reviewPrepare["review"]) > _entryLongest ):
                _entryLongest = len(_reviewPrepare["review"]);
    else:
        raise ValueError("provided preparation mode does not exist");

    return _datasetPrepare, _entryLongest;



#############
# SCORE: (abstract python decimal number) TO LABEL CONVERZZION

const_labelIdMap = {
    "bad": 0,
    "neutral": 1,
    "good": 2,
};

const_idLabelMap = {
    0: "bad",
    1: "neutral",
    2: "good",
};

const_scoreIdMap = {
     0: 0,
     1: 0,
     2: 0,
     3: 0,
     4: 0,
     5: 0,
     6: 1,
     7: 1,
     8: 2,
     9: 2,
    10: 2,
};

def entryLabelConverzzion( _score ):
    return const_scoreIdMap[round(_score)];

def datasetLabelConverzzion( _dataset ):
    _datasetLabel = [];
    for _entry in _dataset:
        _datasetLabel.append( entryLabelConverzzion( _entry ) );
    return _datasetLabel;



#############
# DATASET VECTORIZATION

import numpy;

def entryVectorization( _entry, _model, _fragmentChannels, _entryLongest = -1 ):
    _entryLength = len( _entry );
    if( _entryLongest >= 0 ):
        _currentLength = _entryLongest;
        if( _entryLength > _entryLongest ):
            _entryLength = _entryLongest
    else:
        _currentLength = _entryLength;
    
    _entryVecc = numpy.zeros( (_currentLength, _fragmentChannels), dtype = float );

    for _i in range( _entryLength ):
        _word = _entry[_i];
        if( _word in _model.wv ):
            _wordVecc = _model.wv[_word];
            for _j in range( _fragmentChannels ):
                _entryVecc[_i][_j] = _wordVecc[_j];
    
    return _entryVecc;

def datasetVectorization( _datasetX, _model, _fragmentChannels, _entryLongest = -1 ):
    _datasetVecc = [];
    for _review in _datasetX:
        _reviewVecc = entryVectorization(_review, _model, _fragmentChannels, _entryLongest);
        _datasetVecc.append(_reviewVecc);
    return _datasetVecc;



#############
# DATASET TENSOR CONVERZZION

import tensorflow;

def datasetTensorConverzzion( _datasetX, ):
    _datasetTense = [];
    for _reviewTense in _datasetX:
        _reviewTense = tensorflow.convert_to_tensor(_reviewTense);
        _datasetTense.append(_reviewTense);
    return _datasetTense;

def datasetCategoricalConverzzion( _datasetY ):
    _datasetCat = utils.to_categorical( _datasetY );
    return _datasetCat;
    """_datasetCat = [];
    for _reviewCat in _datasetY:
        _reviewCat = utils.to_categorical(_reviewCat, num_classes = len(const_labelIdMap));
        _datasetCat.append(_reviewCat);
    return _datasetCat;"""




#############
# DOING THE DATA UNTIL ITS WELL DONE

from gensim.models import Word2Vec;
import pandas;
import numpy;
from sklearn.model_selection import train_test_split;
import tensorflow;
from tensorflow.keras import utils;


""" does the data very thoroughly, really properly does it

does data as follows:

dataset: [ entry, ... (n) ];
entry: [ review, score ];
review: [ word, ... (m) ];
word: [ attribute, ... (100) ];
attribute: float32;
score: some_kind_of_abstract_python_decimal_number;

dataset: [ [ [ [ attribute, ... (100) ], ... (m) ], score ] ... (n) ];
datasetX: [ review ... (n) ];
datasetY: [ score ... (n) ];

"""
def doData( _path, _preparationMode = "lemma", _fragmentChannels = 100, _vectorizationMode = "skip", _padVectors = False ):
    # i offer my apologies and regrets for doing this functionally instead of operationally
    # this basically takes twice the memory it needs to take because of it
    # well its ok (ish)
    print("    # started preparation");
    _dataset = pandas.read_csv( _path, usecols = ["Negative_Review", "Positive_Review", "Reviewer_Score"] );
    _dataset = dataframeConverzzion( _dataset );
    _dataset, _entryLongest = datasetPreparation( _dataset, _mode = _preparationMode );
    _datasetX = datasetExtraxxion(_dataset, "review");
    _datasetY = datasetExtraxxion(_dataset, "score");
    _dataset = None;
    print("    # finished preparation");
    
    if( not _padVectors ):
        _entryLongest = -1;

    print("    # started vectorization");
    _model = None;
    if( _vectorizationMode == "skip" ):
        # Create Skip Gram model
        _model = Word2Vec( _datasetX, min_count = 1, vector_size = _fragmentChannels, window = 5, sg = 1 );
    elif( _vectorizationMode == "bow" ):
        # Create CBOW model
        _model = Word2Vec( _datasetX, min_count = 1, vector_size = _fragmentChannels, window = 5 );
    else:
        raise ValueError( "provided vectorization mode does not exist" );
    print("    # finished vectorization");

    _datasetX = datasetVectorization( _datasetX, _model, _fragmentChannels, _entryLongest = _entryLongest );
    
    _datasetY = datasetLabelConverzzion( _datasetY )
    _datasetY = datasetCategoricalConverzzion( _datasetY );

    print("    # dataset size:", len(_datasetX), len(_datasetY));
    
    print("    # started splitting");
    _datasetTrainX, _datasetTestX, _datasetTrainY, _datasetTestY = train_test_split( _datasetX, _datasetY, test_size = 0.20, random_state = 1 );
    print("    # started splitting");

    print("    # outputting dataset");
    return \
        numpy.array(_datasetTrainX), \
        numpy.array(_datasetTestX), \
        _datasetTrainY, \
        _datasetTestY, \
        _entryLongest;
    


#############
# ACTUALLY LOADING UP THE DATA

fragmentChannels = 64;

datasetTrainX, datasetTestX, datasetTrainY, datasetTestY, entryLength = doData("../../Hotel_Reviews_small.csv", _fragmentChannels = fragmentChannels, _padVectors = True );


print(
    "  # # training data: \n",
    "    # textures (number of reviews):\n",
    len(datasetTrainX),
    "    # resolution (fragments per texture):\n",
    len(datasetTrainX[0]),
    "    # colour space dimensionality (channels per fragment):\n",
    len(datasetTrainX[0][0])
);

#############
# NETWORK: DEFINING /// COMPILING /// TRAINING

from matplotlib import pyplot;

import tensorflow;
from tensorflow.keras import layers, models, callbacks;

with tensorflow.device("/GPU:0"):
    model = models.Sequential();
    model.add( layers.Input( shape = ( entryLength, fragmentChannels ), dtype = "float32" ) );
    model.add( layers.Conv1D( 32, 4, activation = "relu", strides = 1, data_format = "channels_last" ) );
    model.add( layers.BatchNormalization() );
    model.add( layers.Dense( 96, activation = "relu" ) );
    model.add( layers.Dropout( 0.2 ) );
    model.add( layers.Flatten() );
    model.add( layers.Dense( 192, activation = "relu" ) );
    model.add( layers.Dropout( 0.3 ) );
    model.add( layers.Dense( 128, activation = "relu" ) );
    model.add( layers.Dropout( 0.2 ) );
    model.add( layers.Dense( 10, activation = "relu" ) );
    model.add( layers.BatchNormalization() );
    model.add( layers.Dense( 3, activation = "relu" ) );

    model.compile(
        optimizer = "adam",
        loss = tensorflow.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics = ["accuracy"],
    );

    early_stopping = callbacks.EarlyStopping( monitor = "val_loss", patience = 20, restore_best_weights = True );

    history = model.fit(
        datasetTrainX,
        datasetTrainY,
        batch_size = 64,
        epochs = 50,
        validation_data = (datasetTestX, datasetTestY),
        callbacks = [early_stopping],
    );

# use flatten layer if youre gonna do several sentences in parrallel



#############
# EVALUATION /// VISUALIZATION
from sklearn.metrics import confusion_matrix, classification_report
import seaborn

def plotAll():
    pyplot.plot( history.history["accuracy"], label = "accuracy" );
    pyplot.plot( history.history["val_accuracy"], label = "val_accuracy" );
    pyplot.xlabel("Epoch");
    pyplot.ylabel("Accuracy");
    pyplot.ylim( [0.5, 1] );
    pyplot.legend( loc = "lower right" );

    pyplot.show();

plotAll();

loss, accuracy = model.evaluate( datasetTestX,  datasetTestY, verbose = 0 );

print( "LOSS", loss, "ACCURACY", accuracy );

predictions = model.predict(datasetTestX);
matrix = confusion_matrix(datasetTestY.argmax(axis=1), numpy.around(predictions, decimals=0).argmax(axis=1));
print(classification_report(numpy.argmax(datasetTestY, axis=1), numpy.argmax(predictions, axis=1)));

conf_matrix = pandas.DataFrame(matrix, index = ['Neutral','Negative','Positive'],columns = ['Neutral','Negative','Positive']);
print(conf_matrix);

# AAAA FIX
conf_matrix = numpy.array(conf_matrix).astype('float') / numpy.array(conf_matrix).sum(axis=1)[:, numpy.newaxis];
pyplot.figure( figsize = (5,5) );
a = seaborn.heatmap( conf_matrix, annot = True, annot_kws = {"size": 15} );
a.imshow();