from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten,  MaxPooling2D, Conv2D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizers import sgd
import matplotlib.pyplot as plt
import pandas as pd
# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255
n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1/6, random_state=1)

print("Size of:")
print("- Training-set:\t\t{}".format(x_train.shape[0]))
print("- Validation-set:\t{}".format(x_val.shape[0]))
print("- Test-set:\t\t{}".format(x_test.shape[0]))
print(" Shape of train target set:{}".format(y_train.shape))

def create_model(learning_rate=0.001, layer_dims=[128, 256, 64]):
    model = Sequential()
    model.add(Conv2D(layer_dims[0], kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(layer_dims[1], kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(layer_dims[2], activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=sgd(lr=learning_rate), metrics=['accuracy'])
    return model

batch_sizes = [128, 256]
learning_rates = [0.05, 0.01]
layer_dims = [[128, 256, 64], [64, 128, 128]]
params = [(batch, alpha, dims) for batch in batch_sizes for alpha in learning_rates for dims in layer_dims]
best_model = None

print("\nHyper-Parameter Search:")
for (batch_size, learning_rate, dims) in params:
    model = create_model(learning_rate, dims)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=10, verbose=0, validation_data=(x_val, y_val))
    print("Batch_size=%d, learning_rate=%f, dims=%s, val-acc=%f" % (batch_size, learning_rate, dims, history.history['val_acc'][-1]))
    if best_model is None or history.history['val_acc'][-1] > best_model[0].history['val_acc'][-1]:
        best_model = (history, model, (batch_size, learning_rate, dims))

history, model, stats = best_model
print("\nBEST MODEL: Batch_size=%d, learning_rate=%f, dims=%s, val-acc=%f" % (*stats, history.history['val_acc'][-1]))
print(pd.DataFrame(history.history))
print(model.summary())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy Vs. Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation Set'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Vs. Epochs')
plt.ylabel('Loss')
plt.legend(['Training Set', 'Validation Set'])
plt.xlabel('Epoch')
plt.show()


result = model.evaluate(x_test, y_test)
print('Test Set Results:')
for name, value in zip(model.metrics_names, result):
    print(name, value)
