
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")                              # load the dataset from the registered tourism.csv inside the data folder
df.drop(columns=["CustomerID","Unnamed: 0"], inplace=True)                        # drop the customer identifier column, it is not a predictive feature

'''
NOTE: categorical columns are intentionally left as raw strings.
The training pipeline one-hot-encodes them, and the Streamlit app also sends
raw category values. Encoding them here (e.g. LabelEncoder) would make training
and serving use different representations, silently breaking predictions.
'''

target = "ProdTaken"                                                              # set `ProdTaken` as target variable
X = df.drop(columns=[target])
y = df[target]

# Split the data
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y                              # stratify keeps the (imbalanced) purchase ratio consistent across splits
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
