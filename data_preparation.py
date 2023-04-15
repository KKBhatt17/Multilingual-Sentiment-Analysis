# Get data
# wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz
# wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

# unzip trainingandtestdata.zip

import pandas as pd

# Train data
df = pd.read_csv("training.1600000.processed.noemoticon.csv",encoding='latin-1',names=['label','id','time','query','user','tweet'])
sentiment_dataframe = df[['tweet','label']]

# Test data
# df = pd.read_csv("data/testdata.manual.2009.06.14.csv",encoding='latin-1',names=['label','id','time','query','user','tweet'])

# Shuffle
sentiment_dataframe = sentiment_dataframe.sample(frac=1)

# Train-Val-Test Split
df1 = sentiment_dataframe.iloc[0:1598000].reset_index(drop=True)
df2 = sentiment_dataframe.iloc[1598000:1599000].reset_index(drop=True)
df3 = sentiment_dataframe.iloc[1599000:1600000].reset_index(drop=True)

df1.to_csv("train.csv")
df2.to_csv("val.csv")
df3.to_csv("test.csv")