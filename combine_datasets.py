import pandas as pd

# Read the CSV files
mat = pd.read_csv('student-mat.csv')
por = pd.read_csv('student-por.csv')

# Define the columns that identify unique students
key_cols = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"
]

# Combine the datasets
combined = pd.concat([mat, por], ignore_index=True)

# Remove duplicate students based on key columns
combined_unique = combined.drop_duplicates(subset=key_cols)

# Print rows and ensure the difference is 382
print("Rows before dropping duplicates:", len(combined))
print("Rows after dropping duplicates:", len(combined_unique))
print("Difference (should be 382):", len(combined) - len(combined_unique))

# Save to a new CSV file
combined_unique.to_csv('dataset.csv', index=False)