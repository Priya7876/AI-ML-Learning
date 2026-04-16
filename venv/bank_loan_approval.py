
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier ,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score 
from sklearn.model_selection import train_test_split
# Create a loan approval dataset
np.random.seed(42)
n_applicants = 1000

loan_data = {
    'Credit_Score': np.random.randint(300, 850, n_applicants),
    'Annual_Income': np.random.randint(20000, 200000, n_applicants),
    'Loan_Amount': np.random.randint(5000, 500000, n_applicants),
    'Employment_Years': np.random.randint(0, 40, n_applicants),
    'Age': np.random.randint(21, 70, n_applicants),
    'Num_Credit_Cards': np.random.randint(0, 10, n_applicants),
    'Num_Bank_Accounts': np.random.randint(1, 8, n_applicants),
    'Debt_to_Income_Ratio': np.random.uniform(0, 0.8, n_applicants),
    'Has_Mortgage': np.random.choice([0, 1], n_applicants),
    'Has_Car_Loan': np.random.choice([0, 1], n_applicants),
}

# Create approval logic
approval = []
for i in range(n_applicants):
    score = 0
    if loan_data['Credit_Score'][i] > 700: score += 3
    if loan_data['Annual_Income'][i] > 75000: score += 2
    if loan_data['Employment_Years'][i] > 5: score += 2
    if loan_data['Debt_to_Income_Ratio'][i] < 0.3: score += 2
    if loan_data['Loan_Amount'][i] < loan_data['Annual_Income'][i] * 3: score += 1
    
    approval.append(1 if score >= 6 else 0)

loan_data['Approved'] = approval
loan_df = pd.DataFrame(loan_data)

print("Loan Approval Dataset:")
print(loan_df.head())
print(f"\nApproval Rate: {loan_df['Approved'].mean()*100:.1f}%")

X = loan_df.drop("Approved" , axis=1)
Y = loan_df["Approved"]
X_Train , X_Test , Y_Train , Y_Test = train_test_split(X,Y ,test_size=0.2 , stratify= Y)


model1 = RandomForestClassifier()
model1.fit(X_Train,Y_Train)
predicted_val_model1 = model1.predict(X_Test)

ac_m1 = accuracy_score(Y_Test , predicted_val_model1)

pd_m1 = precision_score(Y_Test , predicted_val_model1)


model2 = GradientBoostingClassifier()
model2.fit(X_Train,Y_Train)
predicted_val_model2 = model1.predict(X_Test)

ac_m1 = accuracy_score(Y_Test , predicted_val_model2)

pd_m1 = precision_score(Y_Test , predicted_val_model2)


def predict_loan_approval(details):
    input_data = pd.DataFrame(0 , index=0 , columns= details)
    predicted = model2.predict(input_data)[0]
    prob = model1.predict_proba(input_data)

    return predicted,prob


# YOUR TASKS:
# 1. Split data into train/test (80/20)
# 2. Train THREE models:
#    - Decision Tree
#    - Random Forest
#    - XGBoost
# 3. Compare accuracy, precision, recall for each
# 4. Plot ROC curve for XGBoost
# 5. Create a function predict_loan_approval() that:
#    - Takes applicant details
#    - Returns approval decision
#    - Shows probability of approval
#    - Gives suggestions if rejected (e.g., "Improve credit score by 50 points")
# 6. Test with 3 different applicants

# Write your code here