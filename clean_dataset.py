import pandas as pd

dataset = pd.read_csv('dataset.csv')

# Define the columns to be removed
remove_cols = ['school', 'guardian', 'traveltime', 'schoolsup', 'famsup', 'paid', 'nursery', 'internet', 'famrel', 'freetime', 'absences', 'G1', 'G2']

# Drop the specified columns from the dataset
cleaned_dataset = dataset.drop(columns=remove_cols)

# Save the cleaned dataset to a new CSV file
cleaned_dataset.to_csv('cleaned_dataset.csv', index=False)