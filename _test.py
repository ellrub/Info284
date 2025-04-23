import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
import warnings
from gensim.models import Word2Vec;
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models, callbacks
from typing import Iterable
warnings.filterwarnings('ignore')

#############
# DATAFRAME CONVERZZION

import pandas as pd;

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

const_labelMap = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
};

const_LabelUnmap = {
    0: "negative",
    1: "neutral",
    2: "positive",
};

const_scoreMap = {
     0: 0,
     1: 0,
     2: 0,
     3: 0,
     4: 0,
     5: 0,
     6: 0,
     7: 1,
     8: 1,
     9: 2,
    10: 2,
};

def entryLabelConverzzion( _score ):
    return const_scoreMap[round(_score)];

#############
# DATASET SPLITTING / VECTORIZATION

import random;
import numpy as np;

# im doing it like this because it saves memory,
# doing it seperately uses twice as much memory at its peak

def datasetVectorizeSlashSplit( _datasetX, _datasetY, _ratioTest, _channels, _resolution, _categories, _model ):
    _datasetLength = len(_datasetX);
    if( len(_datasetY) != _datasetLength ):
        raise ValueError( "cannot split dataset; provided dataset components do not have the same length." );
        
    _testLength = round(len(_datasetX) * _ratioTest);
    _trainLength = _datasetLength - _testLength;
    
    _trainX = np.zeros((_trainLength, _resolution, _channels), dtype = np.float32);
    _trainY = np.zeros((_trainLength, _categories), dtype = np.float32);
    _testX = np.zeros((_testLength, _resolution, _channels), dtype = np.float32);
    _testY = np.zeros((_testLength, _categories), dtype = np.float32);

    _i = 0;
    _j = 0;
    _k = 0;
    while( _i < _testLength ):
        if( len(_datasetX) <= 0 ):
            break;
        
        _randomIndex = random.randint(0, len(_datasetX) - 1);

        _entry = _datasetX.pop(_randomIndex);
        _entryLength = len(_entry);
        _j = 0;
        while( _j < _resolution and _j < _entryLength ):
            _word = _entry[_j];
            if( _word in _model.wv ):
                _wordVecc = _model.wv[_word];
                _k = 0;
                while( _k < _channels ):
                    _testX[_i][_j][_k] = _wordVecc[_k];
                    _k += 1;
            _j += 1;
        
        _value = _datasetY.pop(_randomIndex);
        _testY[_i][entryLabelConverzzion( _value )] = 1;

        _i += 1;

    _i = 0;
    _j = 0;
    _k = 0;
    while( _i < _trainLength ):
        if( len(_datasetX) <= 0 ):
            break;
        
        _randomIndex = random.randint(0, len(_datasetX) - 1);

        _entry = _datasetX.pop(_randomIndex);
        _entryLength = len(_entry);
        _j = 0;
        while( _j < _resolution and _j < _entryLength ):
            _word = _entry[_j];
            if( _word in _model.wv ):
                _wordVecc = _model.wv[_word];
                _k = 0;
                while( _k < _channels ):
                    _trainX[_i][_j][_k] = _wordVecc[_k];
                    _k += 1;
            _j += 1;

        _value = _datasetY.pop(_randomIndex);
        _trainY[_i][entryLabelConverzzion( _value )] = 1;
    
        _i += 1;
    
    return _trainX, _testX, _trainY, _testY;

import psutil;
import pandas as pd;
import numpy as np;
from gensim.models import Word2Vec;

""" does the data very thoroughly, really properly does it

does data to following format:

n = number of reviews;
m = number of words in review;
k = number of channels per vectorized word

      dataset: [ entry, ... (n) ];
        entry: { "review": texture, "score": socre };
      texture: [ fragment, ... (m) ];
     fragment: [ colourChannel, ... (k) ];
colourChannel: float32;
        score: [ boolean, boolean, boolean ];

dataset: [ [ [ [ float32, ... (k) ], ... (m) ], [ boolean, boolean, boolean ] ] ... (n) ];

datasetX: [ texture ... (n) ];
datasetY: [ score ... (n) ];

"""
def doData( _path, _preparationMode = "lemma", _fragmentChannels = 100, _vectorizationMode = "skip", _padVectors = False ):
    if( _vectorizationMode != "skip" and _vectorizationMode != "bow" ):
        raise ValueError( "provided vectorization mode does not exist" );

    _10thPowerOf2 = 1024;
    _python = psutil.Process();
    print("  >>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10thPowerOf2 * _10thPowerOf2 * _10thPowerOf2)));

    print("  # # started preparation");
    
    print("        # loading from csv");
    _dataset = pd.read_csv( _path, usecols = ["Negative_Review", "Positive_Review", "Reviewer_Score"] );
    print("      >>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10thPowerOf2 * _10thPowerOf2 * _10thPowerOf2)));

    print("        # performing converzzion");
    _dataset = dataframeConverzzion( _dataset );
    _dataset, _resolutionLargest = datasetPreparation( _dataset, _mode = _preparationMode );
    print("      >>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10thPowerOf2 * _10thPowerOf2 * _10thPowerOf2)));

    print("        # performing extraxxion");
    _datasetX = datasetExtraxxion(_dataset, "review");
    _datasetY = datasetExtraxxion(_dataset, "score");
    print("      >>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10thPowerOf2 * _10thPowerOf2 * _10thPowerOf2)));
    _dataset = None;
    print("      >>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10thPowerOf2 * _10thPowerOf2 * _10thPowerOf2)));
    
    print("    # finished preparation");
    
    if( not _padVectors ):
        _resolutionLargest = -1;



    print("  # # started vectorization");
    
    print("        # training vectorization model");
    _model = None;
    if( _vectorizationMode == "skip" ):
        # creates a Skip Gram model
        _model = Word2Vec( _datasetX, min_count = 1, vector_size = _fragmentChannels, window = 5, sg = 1 );
    elif( _vectorizationMode == "bow" ):
        # creates CBOW model
        _model = Word2Vec( _datasetX, min_count = 1, vector_size = _fragmentChannels, window = 5 );
    else:
        raise ValueError( "provided vectorization mode does not exist" );
    print("      >>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10thPowerOf2 * _10thPowerOf2 * _10thPowerOf2)));

    print("        # performing vectorization and splitting");
    _datasetTrainX, _datasetTestX, _datasetTrainY, _datasetTestY = datasetVectorizeSlashSplit( _datasetX, _datasetY, 0.2, _fragmentChannels, _resolutionLargest, len(const_labelMap), _model );
    _datasetX = None;
    _datasetY = None;
    print("      >>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10thPowerOf2 * _10thPowerOf2 * _10thPowerOf2)));


    print("    # finished vectorization");




    print("  # # outputting dataset");
    print("  >>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10thPowerOf2 * _10thPowerOf2 * _10thPowerOf2)));
    return \
        _datasetTrainX, \
        _datasetTestX, \
        _datasetTrainY, \
        _datasetTestY, \
        _resolutionLargest;

#############
# NETWORK: DEFINING /// COMPILING /// TRAINING

import tensorflow as tf;
from tensorflow import keras;
from keras import layers, datasets, models, callbacks;

def makeModel( _datasetTrainX, _datasetTestX, _datasetTrainY, _datasetTestY, _resolution, _channels ):
    print("    # started model");
    with tf.device("/GPU:0"):
        _model = models.Sequential();
        _model.add( layers.Input( shape = ( _resolution, _channels ), dtype = "float32" ) );
        """
        _model.add( layers.Conv1D( 32, 4, activation = "relu", strides = 1, data_format = "channels_last" ) );
        _model.add( layers.BatchNormalization() );
        _model.add( layers.Dense( 96, activation = "relu" ) );
        _model.add( layers.Dropout( 0.2 ) );
        _model.add( layers.Flatten() );
        _model.add( layers.Dense( 192, activation = "relu" ) );
        _model.add( layers.Dropout( 0.3 ) );
        _model.add( layers.Dense( 128, activation = "relu" ) );
        _model.add( layers.Dropout( 0.2 ) );
        _model.add( layers.Dense( 10, activation = "relu" ) );
        _model.add( layers.BatchNormalization() );
        _model.add( layers.Dense( 3, activation = "relu" ) );
        """
        _model.add( layers.Conv1D( 32, 8, activation = "relu", strides = 1, data_format = "channels_last" ) );
        _model.add( layers.Dropout( 0.2 ) );
        _model.add( layers.Dense( 192, activation = "relu" ) );
        _model.add( layers.BatchNormalization() );
        _model.add( layers.Flatten() );
        _model.add( layers.Dropout( 0.3 ) );
        _model.add( layers.Dense( 128, activation = "relu" ) );
        _model.add( layers.Dropout( 0.2 ) );
        _model.add( layers.Dense( 96, activation = "relu" ) );
        _model.add( layers.BatchNormalization() );
        _model.add( layers.Dropout( 0.2 ) );
        _model.add( layers.Dense( 3, activation = "relu" ) );
        _model.compile(
            optimizer = "adam",
            loss = keras.losses.CategoricalCrossentropy( from_logits = True ),
            metrics = ["accuracy"],
        );

        _earlyStopping = callbacks.EarlyStopping( monitor = "val_loss", patience = 20, restore_best_weights = True );

        _history = _model.fit(
            _datasetTrainX,
            _datasetTrainY,
            batch_size = 64,
            epochs = 50,
            validation_data = (_datasetTestX, _datasetTestY),
            callbacks = [_earlyStopping],
        );
    print("    # model training finished");
    return _model, _history;

#############
# EVALUATION /// VISUALIZATION
import matplotlib.pyplot as plt;
from sklearn.metrics import confusion_matrix, classification_report;
import seaborn as sns;

def evaluateModel( _model, _datasetTestX, _datasetTestY ):
    print("  # # evaluating model");
    _loss, _accuracy = _model.evaluate( _datasetTestX,  _datasetTestY, verbose = 0 );

    print( "  >>> LOSS:     " + str(_loss), "\n  >>> ACCURACY: " + str(_accuracy) );

    _predictions = _model.predict( _datasetTestX );
    _matrix = confusion_matrix( _datasetTestY.argmax(axis=1), np.around(_predictions, decimals=0).argmax(axis=1) );
    print(classification_report(np.argmax(_datasetTestY, axis=1), np.argmax(_predictions, axis=1)));

    _confusionMatrix = pd.DataFrame(_matrix, index = ["Negative","Neutral","Positive"], columns = ["Negative","Neutral","Positive"]);
    print(_confusionMatrix);
    _10thPowerOf2 = 1024;
    _python = psutil.Process();
    print("      >>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10thPowerOf2 * _10thPowerOf2 * _10thPowerOf2)));

    return _confusionMatrix;

def plotAll( _history, _confusionMatrix ):
    plt.plot( _history.history["accuracy"], label = "accuracy" );
    plt.plot( _history.history["val_accuracy"], label = "val_accuracy" );
    plt.xlabel("Epoch");
    plt.ylabel("Accuracy");
    plt.ylim( [0.5, 1] );
    plt.legend( loc = "lower right" );

    plt.show();

    _confusionMatrix = np.array(_confusionMatrix).astype("float") / np.array(_confusionMatrix).sum(axis=1)[:, np.newaxis];
    plt.figure( figsize = (5, 5) );
    sns.heatmap( _confusionMatrix, annot = True, annot_kws = {"size": 15} );

#############
# ACTUALLY DOING EVERYTHING

fragmentChannels = 64;

datasetTrainX, datasetTestX, datasetTrainY, datasetTestY, resolution = doData( \
    "./data/Small_Hotel_Reviews.csv", \
    _fragmentChannels = fragmentChannels, \
    _padVectors = True \
);

print(
    "  >>> training data:",
    "\n      >>> textures (number of reviews):",
    "\n            " + str(len(datasetTrainX)),
    "\n      >>> resolution (dimensions of texture in fragments):",
    "\n            " + str(len(datasetTrainX[0])),
    "\n      >>> colour space dimensionality (channels per fragment):",
    "\n            " + str(len(datasetTrainX[0][0]))
);

model, history = makeModel( datasetTrainX, datasetTestX, datasetTrainY, datasetTestY, resolution, fragmentChannels );

confusionMatrix = evaluateModel( model, datasetTestX, datasetTestY );
plotAll( history, confusionMatrix );

## dereferencing for garbage collection
datasetTrainX = None;
datasetTestX = None;
datasetTrainY = None;
datasetTestY = None;
model = None;
history = None;

# OUTPUT
# >>> memory usage: 0.51 gb
#   # # started preparation
#         # loading from csv
#       >>> memory usage: 0.51 gb
#         # performing converzzion
#       >>> memory usage: 0.70 gb
#         # performing extraxxion
#       >>> memory usage: 0.70 gb
#       >>> memory usage: 0.70 gb
#     # finished preparation
#   # # started vectorization
#         # training vectorization model
#       >>> memory usage: 0.70 gb
#         # performing vectorization and splitting
#       >>> memory usage: 0.72 gb
#     # finished vectorization
#   # # outputting dataset
#  >>> memory usage: 0.72 gb
#   >>> training data: 
#       >>> textures (number of reviews): 
#             800 
#       >>> resolution (dimensions of texture in fragments): 
#             185 
#       >>> colour space dimensionality (channels per fragment): 
#             64

#   >>> LOSS:     1.043237566947937 
#   >>> ACCURACY: 0.4699999988079071

#     0       0.21      0.32      0.25        25
#            1       0.00      0.00      0.00        74
#            2       0.53      0.85      0.66       101

#     accuracy                           0.47       200
#    macro avg       0.25      0.39      0.30       200
# weighted avg       0.30      0.47      0.36       200

#           Negative  Neutral  Positive
# Negative        15        0        10
# Neutral         33        0        41
# Positive        48        0        53
#       >>> memory usage: 0.94 gb


# Ny convertion
# import pandas as pd

# def extract_relevant_csv_data( _file_path, _use_cols = ["Negative_Review", "Positive_Review", "Reviewer_Score"], _random_state = 42):
#     _dataset = pd.read_csv(_file_path, usecols = _use_cols)
#     _new_dataset = []
#     for i in _dataset.to_numpy():
#         _new_dataset.append({ "Reviewer_Score": i[2], "Positive_Review": i[0], "Negative_Review" : i[1] })
#     _new_dataset = pd.DataFrame(_new_dataset)

#     if _random_state is not None:
#         _new_dataset = _new_dataset.sample(frac = 1, random_state = _random_state).reset_index(drop = True)
#     else:
#         _new_dataset = _new_dataset.sample(frac = 1).reset_index(drop = True)

#     return _new_dataset
# data = extract_relevant_csv_data('./data/Small_Hotel_Reviews.csv')
# extract_relevant_csv_data('./data/Small_Hotel_Reviews.csv') # To display the data below

# print(max(len(str(entry).split()) for entry in X))

# >>> memory usage: 0.09 gb
#   # # started preparation
#         # loading from csv
#       >>> memory usage: 0.11 gb
#         # performing converzzion
#       >>> memory usage: 0.15 gb
#         # performing extraxxion
#       >>> memory usage: 0.15 gb
#       >>> memory usage: 0.15 gb
#     # finished preparation
#   # # started vectorization
#         # training vectorization model
#       >>> memory usage: 0.16 gb
#         # performing vectorization and splitting
#       >>> memory usage: 0.20 gb
#     # finished vectorization
#   # # outputting dataset
#   >>> memory usage: 0.20 gb
#   >>> training data: 
#       >>> textures (number of reviews): 
#             800 
#       >>> resolution (dimensions of texture in fragments): 
#             185 
#       >>> colour space dimensionality (channels per fragment): 
#             64
# ...
# Negative        12        4         7
# Neutral         26       27        20
# Positive        23       46        35
#       >>> memory usage: 1.49 gb