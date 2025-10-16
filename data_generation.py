import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker()
Faker.seed(42)
np.random.seed(42)

# Ensure directory exists
os.makedirs("data/raw", exist_ok=True)

n = 100000  # 100K transactions
fraud_rate = 0.07  # 7% frauds

transactions = []

# Numeric codes for mapping
merchant_map = {
    0: "Grocery",
    1: "Electronics",
    2: "Travel",
    3: "Clothing",
    4: "Restaurants"
}

channel_map = {
    0: "Online",
    1: "POS",
    2: "ATM"
}

location_map = {
    0: "Mumbai",
    1: "Delhi",
    2: "Bangalore",
    3: "Hyderabad",
    4: "Chennai",
    5: "Kolkata",
    6: "Pune",
    7: "Ahmedabad"
}

for i in range(n):
    amount = round(np.random.exponential(1000), 2)
    is_fraud = 1 if random.random() < fraud_rate else 0

    merchant_code = random.randint(0, len(merchant_map) - 1)
    channel_code = random.randint(0, len(channel_map) - 1)
    location_code = random.randint(0, len(location_map) - 1)

    transactions.append({
        "transaction_id": f"TXN{i+1}",
        "customer_id": f"CUST{random.randint(1000, 9999)}",
        "timestamp": fake.date_time_this_year(),
        "amount": amount,
        "merchant_type": merchant_map[merchant_code],  # ✅ mapped to string
        "channel": channel_map[channel_code],          # ✅ mapped to string
        "location": location_map[location_code],       # ✅ mapped to string
        "device_id": f"DEV{random.randint(100,999)}",
        "is_fraud": is_fraud
    })

df = pd.DataFrame(transactions)

df.to_csv("data/raw/transactions.csv", index=False)
print("Synthetic data generated successfully in data/raw/transactions.csv")
