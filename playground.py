import pandas as pd
from sklearn.model_selection import train_test_split

from configs import training_args

df = pd.read_csv(training_args['train_filepath'])

sentiment_mapping = {'positive': 1, 'negative': 0}
df['label'] = df['sentiment'].map(sentiment_mapping)
df.drop(columns=['sentiment'], inplace=True)

# reduce the dataset to work on an ultralight laptop with 2 GB of Vram
if training_args['reduce_ratio'] is not None:
    df, _ = train_test_split(df, train_size=training_args['reduce_ratio'], shuffle=True, stratify=df['label'])
print(df.head())

# print(f'label 0 count: {(df.label.values == 0).sum()}\nlabel 1 count: {(df.label.values == 1).sum()} ')
unique_labels = sorted(df['label'].unique().tolist())
for label in unique_labels:
    count = (df.label.values == label).sum()
    print(f"label {label} count: {count}")

