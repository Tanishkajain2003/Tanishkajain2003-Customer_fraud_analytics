import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load cleaned data
df = pd.read_csv('data/processed/cleaned_transactions.csv')

# Store original text columns before encoding
original_channel = df['channel']
original_merchant = df['merchant_type']
original_location = df['location']

# Encode for training
df['channel_code'] = df['channel'].astype('category').cat.codes
df['merchant_type_code'] = df['merchant_type'].astype('category').cat.codes
df['location_code'] = df['location'].astype('category').cat.codes

# Features and target
X = df[['amount', 'channel_code', 'merchant_type_code', 'location_code']]
y = df['is_fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Fraud Detection Model Performance:")
print(classification_report(y_test, y_pred))

# Fraud scores
df['fraud_score'] = model.predict_proba(X)[:, 1]

# Restore readable text values
df['channel'] = original_channel
df['merchant_type'] = original_merchant
df['location'] = original_location

# Save final output
df[['transaction_id', 'customer_id', 'timestamp', 'amount',
    'merchant_type', 'location', 'device_id', 'channel',
    'is_fraud', 'fraud_score']].to_csv('data/reports/fraud_predictions.csv', index=False)

print("fraud_predictions.csv saved with readable text columns!")
