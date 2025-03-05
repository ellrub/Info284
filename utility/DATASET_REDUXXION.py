import pandas;
import random;

# REDUCES CSV TO A SMALLER SUBSET OF THE CSV AND MAKES A NEW CSV
def csvReduxxion(_sPath, _sPathNew, _nSizeNew, _nRandomState, _sEncoding = "utf-8"):
    _dataset = pandas.read_csv(_sPath);
    _dataset.sample(n = _nSizeNew, random_state = _nRandomState).to_csv(_sPathNew, encoding = _sEncoding);

# csvReduxxion("../../Hotel_Reviews.csv", "../../Hotel_Reviews_small.csv", 256, 1);