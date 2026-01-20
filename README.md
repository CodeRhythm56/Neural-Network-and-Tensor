

# Neural Network Basics with TensorFlow and Keras

## Activation Functions
An activation function defines the output of a neuron given an input or set of inputs. It introduces non-linearity into the network, which allows it to learn complex patterns.

**Common Types:**
*   **ReLU (Rectified Linear Unit):** The standard function for hidden layers.
    *   Formula: Returns 0 if x is less than 0, and returns x if x is greater than 0.
*   **Sigmoid:** Used for binary classification (Yes/No). It squashes the output between 0 and 1, representing a probability.
*   **Softmax:** Used for multi-class classification. It converts outputs into a probability distribution across all possible classes.

## Model Structure

### Sequential Model
The `Sequential` model is a linear stack of layers. It allows information to flow from one layer to the next in a straight line.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Create the empty shell
model = Sequential()

# 2. Add layers one by one
model.add(Dense(32, input_shape=(10,))) # First layer
model.add(Dense(64))                    # Hidden layer
model.add(Dense(1))                     # Output layer
```

### Dense Layer
This is a standard, fully-connected neural network layer. Every neuron in a `Dense` layer receives input from all neurons in the previous layer.

**Parameters:**
*   `units`: The number of neurons in the layer.
*   `input_shape`: Required for the first layer only. It specifies the shape of the input data (e.g., `(10,)` for data with 10 features).
*   `activation`: The activation function to use (e.g., `'relu'`).

```python
# Create a layer with 64 neurons, using ReLU activation
layer = Dense(units=64, activation='relu')
```

### Flatten Layer
The `Flatten` layer transforms multi-dimensional input (such as a 2D image matrix) into a one-dimensional array (a vector).

*   **Use case:** Typically used as the first layer when processing images to unroll pixel data.
*   **Example:** Converts a `28x28` image into a vector of `784` pixels.

```python
from tensorflow.keras.layers import Flatten, Dense

model = Sequential()
# Takes a 28x28 image and flattens it to 784 numbers
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
```

## Compilation

Before training, the model must be compiled with three key components via `model.compile(optimizer, loss, metrics)`.

1.  **Optimizer**: The algorithm that adjusts the weights of the network to minimize the loss.
    *   *Recommendation:* Use `'adam'`. It is the industry standard and adapts the learning rate during training.
2.  **Loss Function**: Calculates how inaccurate the model is. The model tries to minimize this number.
    *   `mse`: Mean Squared Error. Used for Regression (predicting continuous numbers like prices).
    *   `binary_crossentropy`: Used for Binary Classification (two classes).
    *   `categorical_crossentropy`: Used for Multi-Class Classification (three or more classes).
3.  **Metrics**: Used to monitor the training and testing steps.
    *   *Common usage:* `['accuracy']`.

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

## Training and Evaluation

### Training the Model
The `model.fit()` function trains the model for a fixed number of iterations (epochs) on the dataset.

*   `x`: The training data (features).
*   `y`: The training answers (labels).
*   `epochs`: The number of times the model works through the entire dataset.
*   `batch_size`: The number of samples processed before the model is updated. Default is usually 32.

```python
# Train the model for 10 rounds
model.fit(x_train, y_train, epochs=10)
```

### Evaluating the Model
The `model.evaluate()` function returns the loss value and metrics values for the model in test mode.

```python
# Check how accurate the model is
loss, accuracy = model.evaluate(x_test, y_test)
```

### Making Predictions
The `model.predict()` function generates output predictions for new input samples.

```python
# Make a prediction on a new data point
prediction = model.predict(new_data)
```
