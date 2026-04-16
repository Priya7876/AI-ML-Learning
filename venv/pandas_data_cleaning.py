import pandas as pd

df = pd.read_csv("C:\\Users\\priya\\Downloads\\dirty_medical_data.csv")

# Step 1: Drop nulls
df = df.dropna()

# Step 2: Clean text
df["symptoms"] = df["symptoms"].str.strip().str.lower()

# Step 3: Split symptoms into list
df["symptoms"] = df["symptoms"].str.split(",")

# Step 4: Clean each symptom
df["symptoms"] = df["symptoms"].apply(
    lambda x: [i.strip() for i in x if i.strip() != ""]
)

print(df.head())
# Step 5: Create unique symptom list
symptoms_set = set()

for row in df["symptoms"]:
    symptoms_set.update(row)

print(symptoms_set)
sym_list = list(symptoms_set)

print("Unique symptoms:", sym_list)


# Step 6: Create feature matrix
X = pd.DataFrame(0, index=range(len(df)), columns=sym_list)

# Step 7: Fill matrix
for i in range(len(df)):
    for symptom in df.loc[i, "symptoms"]:
        X.loc[i, symptom] = 1

print(X.head())