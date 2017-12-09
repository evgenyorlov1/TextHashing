#!/usr/bin/env python
import pickle

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

print 'Begin TF-IDF vectorization'
# The raw text dataset is stored as tuple in the form:
# (X_train_raw, y_train_raw, X_test_raw, y_test)
# The 'filtered' dataset excludes any articles that we failed to retrieve
# fingerprints for.
raw_text_dataset = pickle.load(open("data/raw_text_dataset.pickle", "rb" ))
X_train_raw = raw_text_dataset[0]
y_train_labels = raw_text_dataset[1]
X_test_raw = raw_text_dataset[2]
y_test_labels = raw_text_dataset[3]

# The Reuters dataset consists of ~100 categories. However, we are going to
# simplify this to a binary classification problem. The 'positive class' will
# be the articles related to "acquisitions" (or "acq" in the dataset). All
# other articles will be negative.
y_train = ["acq" in y for y in y_train_labels]
y_test = ["acq" in y for y in y_test_labels]

print("  %d training examples (%d positive)" % (len(y_train), sum(y_train)))
print("  %d test examples (%d positive)" % (len(y_test), sum(y_test)))


# Tfidf vectorizer:
#   - Strips out stop words
#   - Filters out terms that occur in more than half of the docs (max_df=0.5)
#   - Filters out terms that occur in only one document (min_df=2).
#   - Selects the 10,000 most frequently occuring words in the corpus.
#   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of
#     document length on the tf-idf values.
vectorizer = TfidfVectorizer(max_df=0.5,
                             max_features=1000,
                             min_df=2,
                             stop_words='english',
                             use_idf=True)

# Build the tfidf vectorizer from the training data ("fit"), and apply it
# ("transform").
X_train_tfidf = vectorizer.fit_transform(X_train_raw)
# Apply todense() to retrieve dense matrix of(n_samples, n_features)
X_train_tfidf = X_train_tfidf.todense()

# Now apply the transformations to the test data as well.
X_test_tfidf = vectorizer.transform(X_test_raw)
X_test_tfidf = X_test_tfidf.todense()
print 'X_test_tfidf.shape {0}'.format(np.shape(X_test_tfidf)) # (4858, 10000)


print 'Begin AutoEncoder building'
from keras.layers import Input, Dense, Activation
from keras.models import Model

encoding_dim = 256  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
input_dim = np.shape(X_train_tfidf)[1]

# this is our input placeholder
input_txt = Input(shape=(1000,))
# "encoded" is the encoded representation of the input
encoded = Dense(256, activation='relu')(input_txt)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(1000, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_txt, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_txt, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(256,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(X_train_tfidf, X_train_tfidf,
                epochs=20,
                batch_size=256,
                shuffle=True,
                validation_data=(X_train_tfidf, X_train_tfidf))

encoded_train_txt = encoder.predict(X_train_tfidf)


print 'Begin kNN classification'
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
knn.fit(encoded_train_txt, y_train)

encoded_test_txt = encoder.predict(X_test_tfidf)

p = knn.predict(encoded_test_txt)

numRight = 0
for i in range(0,len(p)):
    if p[i] == y_test[i]: numRight += 1

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))