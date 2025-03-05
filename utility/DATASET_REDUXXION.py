import pandas;
import random;

# REDUCES CSV TO A SMALLER SUBSET OF THE CSV AND MAKES A NEW CSV
def csvReduxxion(_sPath, _sPathNew, _nSizeNew, _nRandomState, _sEncoding = "utf-8"):
    _dataset = pandas.read_csv(_sPath);
    _dataset.sample(n = _nSizeNew, random_state = _nRandomState).to_csv(_sPathNew, encoding = _sEncoding);

<<<<<<< HEAD
# csvReduxxion("../../Hotel_Reviews.csv", "../../Hotel_Reviews_small.csv", 256, 1);
=======
csvReduxxion("data/Hotel_Reviews.csv", "data/DATASET_REDUX.csv", 1000, random.randint(0, 1000));
>>>>>>> 79c6db82a348693796054aa7803e8d4c6c7e0aec
