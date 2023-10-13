import tensorflow as tf
from tensorflow import keras
from pre import generating_training_sequences, SEQUENCE_LENGTH
OUTPUT_UNITS = 25
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL = "Model/model.h5"
def build_model(output_units,num_units,loss,learning_rate):
    input = keras.layers.Input(shape=(None,output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(output_units,activation="softmax")(x)
    model = keras.Model(input, output)
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=["accuracy"])
    model.summary()
    return model
def train(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,learning_rate=LEARNING_RATE):
    inputs,targets = generating_training_sequences(SEQUENCE_LENGTH)
    model = build_model(output_units,num_units ,loss,learning_rate)
    model.fit(inputs,targets,epochs=EPOCHS,batch_size=BATCH_SIZE)
    model.save(SAVE_MODEL)
if __name__ == "__main__":
    train()