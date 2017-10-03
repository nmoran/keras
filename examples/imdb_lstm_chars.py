'''Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np

maxlen = 500  # cut texts after this number of chars
batch_size = 32

def get_alphabet(phrases):
    """
    Get unique letters from array of phrases

    Parameters
    ----------
    phrases: string array 
        Array of strings

    Return
    -------
    array:
        Unique characters reverse sorted by frequency
    array:
        Corresponding frequencies
    """
    chars, counts = np.unique(np.hstack([[y for y in x] for x in phrases]), return_counts=True)
    idxs = np.argsort(counts)[::-1]
    return chars[idxs], counts[idxs]

def indices_to_phrase(idxs, rwi):
    return " ".join([(' ' if x < 4 else rwi[x-3]) for x in idxs[1:]])

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data()

# get the actual words
wi = imdb.get_word_index()
rwi = dict([[wi[x], x] for x in wi])
all_train = np.concatenate((x_train, x_test))
all_train_words = [indices_to_phrase(x, rwi) for x in all_train]
alphabet, freq = get_alphabet(all_train_words)
alphabet_dict = dict((y, x) for x, y in enumerate(alphabet))
max_features = len(alphabet)

x_train = [np.array([alphabet_dict[x] for x in indices_to_phrase(y, rwi)]) for y in x_train]
x_test = [np.array([alphabet_dict[x] for x in indices_to_phrase(y, rwi)]) for y in x_test]
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
