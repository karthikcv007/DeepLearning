import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# -------------------------------
# 1. Load dataset
# -------------------------------

df = pd.read_csv("/content/monthly_csv.csv")

print("Dataset columns:")
print(df.columns)

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Sort according to date
df = df.sort_values("Date")

# Use Mean column
temperature = pd.to_numeric(
    df["Mean"],
    errors="coerce"
).values

# Remove missing values
temperature = temperature[~np.isnan(temperature)]

print("\nNumber of values:", len(temperature))

# -------------------------------
# 2. Normalize the data
# -------------------------------

minimum = temperature.min()
maximum = temperature.max()

data = (temperature - minimum) / (maximum - minimum)

# -------------------------------
# 3. Create input sequences
# -------------------------------

sequence_length = 5

X = []
y = []

for i in range(len(data) - sequence_length):

    # Previous 5 months
    X.append(data[i:i + sequence_length])

    # Next month
    y.append(data[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Reshape for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

# -------------------------------
# 4. Build LSTM model
# -------------------------------

model = tf.keras.Sequential([

    tf.keras.Input(
        shape=(sequence_length, 1)
    ),

    tf.keras.layers.LSTM(50),

    tf.keras.layers.Dense(1)
])

# -------------------------------
# 5. Compile model
# -------------------------------

model.compile(
    optimizer="adam",
    loss="mse"
)

# -------------------------------
# 6. Train the model
# -------------------------------

model.fit(
    X,
    y,
    epochs=20,
    batch_size=32,
    verbose=1
)

# -------------------------------
# 7. Predict
# -------------------------------

predicted = model.predict(
    X,
    verbose=0
)

predicted = predicted.flatten()

# Convert back to original values
predicted = (
    predicted * (maximum - minimum)
    + minimum
)

actual = (
    y * (maximum - minimum)
    + minimum
)

# -------------------------------
# 8. Plot Actual vs Predicted
# -------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    actual,
    label="Actual Mean"
)

plt.plot(
    predicted,
    label="Predicted Mean"
)

plt.title("Actual vs Predicted Mean Temperature")

plt.xlabel("Months")

plt.ylabel("Mean Temperature")

plt.legend()

plt.show()
