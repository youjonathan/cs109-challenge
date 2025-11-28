import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('cleaned_dataset.csv')

# dummy encoding for binary nominal features
binary_maps = {
    'sex': {'F': 0, 'M': 1},
    'address': {'U': 0, 'R': 1},
    'famsize': {'LE3': 0, 'GT3': 1},
    'Pstatus': {'T': 0, 'A': 1},
    'activities': {'no': 0, 'yes': 1},
    'romantic': {'no': 0, 'yes': 1}
}

for col, mapping in binary_maps.items():
    df[col] = df[col].map(mapping)

# one-hot encoding for multi-categorical nominal features
one_hot_features = ['Mjob', 'Fjob']
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(df[one_hot_features])
encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder.get_feature_names_out(one_hot_features)
)

# concatenate the original dataframe with the encoded features and drop the original columns
df = pd.concat([df.drop(columns=one_hot_features), encoded_df], axis=1)

# drop and save target so it's last column
target = df.pop('G3')
df['G3'] = target

# save the encoded dataset
df.to_csv("encoded_dataset.csv", index=False)

