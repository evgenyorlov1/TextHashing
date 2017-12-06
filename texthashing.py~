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

# Tfidf vectorizer:
#   - Strips out stop words
#   - Filters out terms that occur in more than half of the docs (max_df=0.5)
#   - Filters out terms that occur in only one document (min_df=2).
#   - Selects the 10,000 most frequently occuring words in the corpus.
#   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of
#     document length on the tf-idf values.
vectorizer = TfidfVectorizer(max_df=0.5,
                             max_features=10000,
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


from keras.layers import Input, Dense
from keras.models import Model


encoding_dim = 128  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
input_dim = np.shape(X_train_tfidf)[1]

# this is our input placeholder
input_txt = Input(shape=(10000,))
# "encoded" is the encoded representation of the input
encoded = Dense(128, activation='relu')(input_txt)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(10000, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_txt, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_txt, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(128,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(X_train_tfidf, X_train_tfidf,
                epochs=1,
                batch_size=256,
                shuffle=True,
                validation_data=(X_train_tfidf, X_train_tfidf))

encoded_txt = encoder.predict(X_test_tfidf)
print 'encoded_txt shape {0}'.format(np.shape(encoded_txt))
print 'encoded_txt[0] {0}'.format(encoded_txt[0])
decoded_txt = decoder.predict(encoded_txt)
print 'decoded_txt shape {0}'.format(np.shape(decoded_txt))
print 'decoded_txt[0] {0}'.format(decoded_txt[0])
