pip install nltk tensorflow

import os
import numpy as np
import random
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
import nltk

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and load intents file
lemmatizer = WordNetLemmatizer()

with open('/content/intents.json') as file:
    data = json.load(file)

# Preprocessing the data
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for w in words:
        bag.append(1) if w in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle the features and separate into train_x and train_y
random.shuffle(training)
train_x = np.array([np.array(x[0]) for x in training])
train_y = np.array([np.array(x[1]) for x in training])

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train and save the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=0)  # Suppress progress bar
model.save('chatbot_model.h5')

print("Model created successfully!")

# Chat with the bot
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

def predict_class_silent(sentence, model):
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        p = bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
    return res

def predict_class(sentence, model):
    p = predict_class_silent(sentence, model)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(p) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Starting the chat
print("Chatbot is ready to chat! Type 'quit' to exit.")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    ints = predict_class(message, model)
    res = get_response(ints, data)
    print(f"Bot: {res}")
