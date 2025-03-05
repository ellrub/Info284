import pandas;

def csvConverzzion(_Path, _sPathNew, _useCols = ["Negative_Review", "Positive_Review", "Reviewer_Score"], _sEncoding = "utf-8"):
    _dataset = pandas.read_csv(_Path, usecols = _useCols);
    _mergedReviews = [];
    for i in _dataset.to_numpy():
        _mergedReviews.append([ i[2], i[0] + i[1] ]);
    pandas.DataFrame(_mergedReviews).to_csv(_sPathNew, encoding = _sEncoding);


def dataframeToCsvConverzzion(_dataset, _sPathNew, _sEncoding = "utf-8"):
    _mergedReviews = [];
    for i in _dataset.to_numpy():
        _mergedReviews.append([ i[2], i[0] + i[1] ]);
    pandas.DataFrame(_mergedReviews).to_csv(_sPathNew, encoding = _sEncoding);


def dataframeConverzzion(_dataset):
    _mergedReviews = [];
    for i in _dataset.to_numpy():
        _mergedReviews.append([ i[2], i[0] + i[1] ]);
    return pandas.DataFrame(_mergedReviews);

# dataset = pandas.read_csv("../../Hotel_Reviews_small.csv", usecols=["Negative_Review", "Positive_Review", "Reviewer_Score"]);
# csvTextMergeThing(dataset, "../../Hotel_Reviews_smaller_very_smol.csv");
