import pandas as pd
from sklearn.model_selection import train_test_split

from configs import training_args

# Load your CSV file into a DataFrame
df = pd.read_csv(r'datasets/imdb_binary/IMDB Dataset.csv')

# Assuming 'review' is the feature and 'sentiment' is the target column
X = df['review']
y = df['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

train_df = pd.DataFrame({'review': X_train, 'sentiment': y_train})
test_df = pd.DataFrame({'review': X_test, 'sentiment': y_test})

print('=' * 20, 'train_dataset', '=' * 20)
print(train_df.describe())
print('=' * 20, 'test_dataset', '=' * 20)
print(test_df.describe())

# Save the DataFrames to separate CSV files
train_df.to_csv(r'datasets/imdb_binary/train.csv', index=False)
test_df.to_csv(r'datasets/imdb_binary/test.csv', index=False)
