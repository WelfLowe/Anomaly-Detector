import pandas as pd
from sklearn.preprocessing import StandardScaler

# Path to your file
file_path = "training_log.txt"

# Read the text file and parse it into a DataFrame
data = []
with open(file_path, "r") as file:
    for line in file:
        # Split on commas first, then split each key-value pair on the colon
        parsed_line = {
            key.strip():
            float(value.strip())
            if key.strip() in ['Loss', 'Label'] else int(value.strip())
            for key, value in (item.split(":") for item in line.split(", "))
        }
        data.append(parsed_line)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)


# Normalize the 'Loss' column within each (Epoch, Batch) group
def normalize_epoch_batch(group_df):
    scaler = StandardScaler()
    group_df['Normalized_Loss'] = scaler.fit_transform(group_df[['Loss']])
    return group_df


# Apply normalization to each (Epoch, Batch) group
df = df.groupby(['Epoch', 'Batch'],
                group_keys=False).apply(normalize_epoch_batch)

# Save or inspect the resulting DataFrame
print(df)

# Optionally save it to a CSV file
df.to_csv("training_log.csv", index=False)
