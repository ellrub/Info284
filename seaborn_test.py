import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("data/DATASET_REDUX.csv")
sns.pairplot(df, hue = 'Average_Score')
plt.show()