import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# We will teach the model with 6 examples.
# Features: [Square Footage, Number of Bedrooms]
X_train = np.array([
    [1000, 1],  # Small apartment
    [1200, 2],  # Medium apartment
    [1500, 3],  # Small house
    [2000, 3],  # Medium house
    [2500, 4],  # Large house
    [3000, 5]   # Mansion
], dtype=float)

# Labels: The Price (in thousands, e.g., 200 = $200,000)
y_train = np.array([
    150,  # Price for 1000sqft, 1bed
    200,  # Price for 1200sqft, 2bed
    280,  # Price for 1500sqft, 3bed
    350,  # Price for 2000sqft, 3bed
    450,  # Price for 2500sqft, 4bed
    600   # Price for 3000sqft, 5bed
], dtype=float)


# 2. BUILD THE MODEL

model = Sequential()

# Input Layer & Hidden Layer 1 both in same line (it can be seperate if you want)
# input_dim=2 because we have 2 inputs (sqft and bedrooms)
# We use 'relu' activation for the hidden layers
model.add(Dense(64, input_dim=2, activation='relu'))

# Hidden Layer 2 (Deep learning)
model.add(Dense(32, activation='relu'))

# Output Layer
# We only want 1 number output: The Price.
# We use NO activation function (linear) because we want to predict a raw number.
model.add(Dense(1))


# 3. COMPILE THE MODEL

# Loss='mean_squared_error' is standard for predicting numbers (i.e Regression)
# Optimizer='adam' is the standard efficient solver
model.compile(optimizer='adam', loss='mean_squared_error')


# 4. TRAIN THE MODEL

print("Training the model...")
# epochs=2000 means it loops over the data 2000 times to get smarter
# verbose=0 means "don't print the progress bar for every single epoch"
model.fit(X_train, y_train, epochs=2000, verbose=0)
print("Training finished\n")


# 5. PREDICT NEW DATA

# Let's ask the model to predict prices for houses it hasn't seen before.
X_new = np.array([
    [1800, 3],  # A house with 1800 sqft and 3 bedrooms
    [4000, 5]   # A massive house with 4000 sqft and 5 bedrooms
])

print("Making predictions on new data:")
predictions = model.predict(X_new)

for i, pred in enumerate(predictions):
    sqft = X_new[i][0]
    beds = X_new[i][1]
    print(f"House {i+1}: {sqft} sqft, {beds} beds -> Predicted Price: ${pred[0]:.2f}k")




"""
OUTPUT

Training the model...
Training finished

Making predictions on new data:
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step
House 1: 1800 sqft, 3 beds -> Predicted Price: $332.47k
House 2: 4000 sqft, 5 beds -> Predicted Price: $745.07k

"""
