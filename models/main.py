from numpy import asarray, zeros
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten, Embedding, Dense, Dropout, Conv1D, Conv2D
from keras.utils import to_categorical
from keras.models import Sequential
from file_reader import FileReader

# ------------ Hyperparameters go here ------

EPOCHS = 5
BATCH_SIZE = 10000
# Has to be the same as the GloVe vector dimension (look at file name)
EMBEDDING_DIM = 100

# ------------ Hyperparameters end here -----

# TODO: Add test files as well and test it on that instead of validation
# TODO: Decouple the GloVe and pre-processing stuff from the model creation
# TODO: Create different model files for Dense Network, CNN and RNN
# Step 1: Get the datasets (already split into training, validation and test sets)
reader = FileReader()
reader.read_from_file()

all_inputs, all_labels = reader.return_all_data()
training_inputs, training_labels = reader.return_training_sets()
valid_inputs, valid_labels = reader.return_valid_sets()

# Step 2.1: Label pre-processing
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
encoded_training_labels = label_encoder.transform(training_labels)
encoded_valid_labels = label_encoder.transform(valid_labels)
categorical_training_labels = to_categorical(encoded_training_labels)
categorical_valid_labels = to_categorical(encoded_valid_labels)
print("Fitting the tweets into the following classes: " + str(label_encoder.classes_))

num_classes = str(len(label_encoder.classes_))

# Step 2.2: Prepare and use tokenizer
print("Creating token dictionary...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_inputs)
vocabulary_size = len(tokenizer.word_index) + 1
print("Vocabulary size is: " + str(vocabulary_size))

# Step 2.3: Encode the documents
print("Encoding the input files...")
encoded_training_inputs = tokenizer.texts_to_sequences(training_inputs)
encoded_valid_inputs = tokenizer.texts_to_sequences(valid_inputs)

# Step 2.4: Padding the tweet length to a maximum of 140 characters
max_length = len(max(encoded_training_inputs, key=len))
print("Maximum tweet length is: " + str(max_length))
print("Padding all input sequences to this value...")
padded_training_inputs = pad_sequences(encoded_training_inputs, maxlen=max_length, padding='post')
padded_valid_inputs = pad_sequences(encoded_valid_inputs, maxlen=max_length, padding='post')

# Step 3: Loading the GloVe word embedding file
print("Loading GloVe file...")
embeddings_index = dict()
embedding_file = open('./data/glove.6B.100d.txt')
for line in embedding_file:
    values = line.split()
    embeddings_index[values] = asarray(values[1:], dtype='float32')
embedding_file.close()
print("Loaded " + str(len(embeddings_index)) + " word vectors.")

# Step 4: Creating embedding matrix
print("Creating embedding matrix...")
embedding_matrix = zeros((vocabulary_size, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Step 5: Creating the model
print("Creating the model...")
model = Sequential()
print("Adding the layers...")
model.add(Embedding(vocabulary_size,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=max_length,
                    trainable=False))
model.add(Flatten())

model.add(Dense(14, activation='softmax'))

print("Compiling the model with the optimizer and loss function")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

print("Training...")
model.fit(padded_training_inputs,
          categorical_training_labels,
          validation_data=(padded_valid_inputs, categorical_valid_labels),
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=2)

# evaluate the model
loss, accuracy = model.evaluate(padded_valid_inputs, categorical_valid_labels, verbose=2)
print("Accuracy: " + str(accuracy*100))
