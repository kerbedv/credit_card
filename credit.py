import seaborn
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import codecademylib3

# Load the data
transactions = pd.read_csv('transactions_modified.csv')
print(transactions.head())
print(transactions.info())

# How many fraudulent transactions?
print(transactions['isFraud'].sum())

# Summary statistics on amount column
print(transactions['amount'].describe())

# Create isPayment field
transactions['isPayment'] = 0
transactions.loc[transactions['type'].isin(['DEBIT', 'PAYMENT']), 'isPayment'] = 1

# Create isMovement field
transactions['isMovement'] = 0
transactions.loc[transactions['type'].isin(['CASH_OUT', 'TRANSFER']), 'isMovement'] = 1

# Create accountDiff field
transactions['accountDiff'] = abs(transactions['oldbalanceDest'] - transactions['oldbalanceOrg'])

# Create features and label variables
features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]
label = transactions['isFraud']

# Split dataset
features_train, features_test, label_train, label_test = train_test_split(features, label, test_size = 0.3)

# Normalize the features variables
regulate = StandardScaler()
features_train = regulate.fit_transform(features_train)
features_test = regulate.transform(features_test)


# Fit the model to the training data
log = LogisticRegression()
log.fit(features_train, label_train)

# Score the model on the training data
print(log.score(features_train, label_train))

# Score the model on the test data
print(log.score(features_test, label_test))

# Print the model coefficients
print(log.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
your_transaction = np.array([10000000.31, 0.0, 1.0, 51.5])

# Combine new transactions into a single array
sample_transactions = np.stack([transaction1, transaction2, transaction3, your_transaction])

# Normalize the new transactions
sample_transactions = regulate.transform(sample_transactions)

# Predict fraud on the new transactions
print(log.predict(sample_transactions))

# Show probabilities on the new transactions
print(log.predict_proba(sample_transactions))
