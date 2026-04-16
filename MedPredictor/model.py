from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# =========================
# LOAD DATA
# =========================
DTSDf = pd.read_csv("C:\\Users\\priya\\Downloads\\symtomsToDisease\\DiseaseAndSymptoms.csv")
DTDDf = pd.read_csv("C:\\Users\\priya\\Downloads\\diseasetoDrug\\diseaseToDrug.csv")

# =========================
# CLEAN DATA
# =========================
DTSDf = DTSDf.fillna("")
DTSDf = DTSDf.drop_duplicates()
DTSDf = DTSDf.reset_index(drop=True)
# Normalize symptoms (IMPORTANT)
for col in DTSDf.columns[1:]:
    DTSDf[col] = (
        DTSDf[col]
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

# Normalize disease names in drug file
DTDDf["disease"] = DTDDf["disease"].str.strip().str.lower()

# =========================
# CREATE FEATURE LIST
# =========================
symptoms_set = set()

for col in DTSDf.columns[1:]:
    symptoms_set.update(DTSDf[col].unique())

symptoms_set.discard("")
sym_list = list(symptoms_set)

# =========================
# CREATE FEATURE MATRIX
# =========================
X = pd.DataFrame(0, index=np.arange(len(DTSDf)), columns=sym_list)

for i in range(len(DTSDf)):
    for col in DTSDf.columns[1:]:
        symptom = DTSDf.loc[i, col]
        if symptom != "":
            X.loc[i, symptom] = 1

# Target
Y = DTSDf["Disease"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)

model.fit(X_Train, Y_Train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_Test)

print("Train Accuracy:", model.score(X_Train, Y_Train))
print("Test Accuracy:", model.score(X_Test, Y_Test))
print("Final Accuracy:", accuracy_score(Y_Test, y_pred))

# =========================
# DRUG MAP
# =========================
drug_map = DTDDf.groupby("disease")["drug"].apply(list).to_dict()

# =========================
# TOP-K PREDICTION FUNCTION
# =========================
def predict_top_k(symptoms_input, k=3):
    # Normalize input
    symptoms_input = [
        s.strip().lower().replace(" ", "_") for s in symptoms_input
    ]

    input_data = pd.DataFrame(0, index=[0], columns=X.columns)

    valid_symptoms = []
    invalid_symptoms = []

    for sym in symptoms_input:
        if sym in input_data.columns:
            input_data[sym] = 1
            valid_symptoms.append(sym)
        else:
            invalid_symptoms.append(sym)

    print("Used Symptoms:", valid_symptoms)
    if invalid_symptoms:
        print("Ignored Symptoms:", invalid_symptoms)

    # Edge case
    if len(valid_symptoms) == 0:
        print("No valid symptoms provided!")
        return []

    probs = model.predict_proba(input_data)[0]
    top_k_idx = np.argsort(probs)[-k:][::-1]

    results = [(model.classes_[i], probs[i]) for i in top_k_idx]

    return results

# =========================
# GET DRUGS
# =========================
def get_drugs(disease):
    return drug_map.get(disease.strip().lower(), ["No drug found"])

# =========================
# TEST INPUT
# =========================
symptoms = [
    "yellowing of eyes",
    "vomiting",
    "fatigue",
    "nausea",
    "loss of appetite",
    "abdominal pain"
]

# =========================
# RUN
# =========================
top_predictions = predict_top_k(symptoms)

print("\nTop Predictions:")
for disease, prob in top_predictions:
    print(f"{disease} -> {prob:.4f}")

if top_predictions:
    best_disease = top_predictions[0][0]
    drugs = get_drugs(best_disease)

    print("\nPredicted Disease:", best_disease)
    print("Recommended Drugs:", drugs)