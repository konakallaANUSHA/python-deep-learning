import pandas as pd


train_df = pd.read_csv('train.csv')

print(train_df)

train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



print(train_df['Survived'].corr(train_df['Sex']))

#correlation is the statistical summary of the relationship
# between variables and how to calculate it for different types variables and relationships.