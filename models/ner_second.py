import numpy as np
import pandas as pd

from subprocess import check_output

from tensorflow_core.python.keras.models import Model

print(check_output(["ls", "/Users/sulabhkothari/Documents/entity-annotated-corpus/ner.csv"]).decode("utf8"))
dframe = pd.read_csv("/Users/sulabhkothari/Documents/entity-annotated-corpus/ner.csv", encoding="ISO-8859-1",
                     error_bad_lines=False)

dataset = dframe.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
                       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
                       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
                       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
                       'prev-prev-word', 'prev-shape', 'prev-word', "pos"], axis=1)
dataset.info()
dataset.head()
dataset = dataset.drop(['shape'], axis=1)
dataset.head()

class SentenceGetter(object):

    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),
                                                     s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter = SentenceGetter(dataset)
sentences = getter.sentences
print(sentences[5])
maxlen = max([len(s) for s in sentences])
print('Maximum sequence length:', maxlen)

# import matplotlib.pyplot as plt
# #%matplotlib inline
# plt.style.use("ggplot")
# plt.hist([len(s) for s in sentences], bins=50)
# plt.show()

words = list(set(dataset["word"].values))
words.append("ENDPAD")
n_words = len(words)
tags = list(set(dataset["tag"].values))
n_tags = len(tags)

word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

print(word2idx['Obama'])
print(tag2idx["O"])

new_sent = np.pad(np.array(
    [word2idx["I"], word2idx["was"], word2idx["visiting"], word2idx["New"], word2idx["Brighton"], word2idx["during"]
        , word2idx["prime"], word2idx["time"], word2idx["for"], word2idx["conference"], word2idx["and"], word2idx["recruitment"]]), [0, 128])
print("NEWSENTENCE")
print(new_sent)
print(new_sent.shape)

from tensorflow.keras.preprocessing.sequence import pad_sequences

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=140, sequences=X, padding="post", value=n_words - 1)
y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=140, sequences=y, padding="post", value=tag2idx["O"])
from tensorflow.keras.utils import to_categorical

y = [to_categorical(i, num_classes=n_tags) for i in y]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import tensorflow.keras.layers as layers
print(X_test[0].shape)
input = layers.Input(shape=(140,))
model = Embedding(input_dim=n_words, output_dim=140, input_length=140)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer

model = Model(input, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#p = model.predict(np.array([X_test[0]]))
history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=1, validation_split=0.2, verbose=1)

i = 0
p = model.predict(np.array([new_sent]))
p = np.argmax(p, axis=-1)
print("{:14} ({:5}): {}".format("Word", "True", "Pred"))
for w, pred in zip(new_sent, p[0]):
    print("{:14}: {}".format(words[w], tags[pred]))

# https://www.kaggle.com/navya098/bi-lstm-for-ner
