#encoding=utf-8
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import string
import json
import jieba
import time


from keras import backend
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

with open('data/question_labels.json', 'r') as f:
    question_labels = json.load(f)


q_dict = {'NUMBER': 0, 'PERSON': 1, 'LOCATION': 2, 'ORGANIZATION': 3, 'ARTIFACT': 4, 'TIME': 5, 'PROCEDURE': 6, 'AFFIRMATION': 7, 'CAUSALITY': 8}

q_zh = []
q_type = []
for line in question_labels:
    q_zh.append(line['q_zh'])
    q_type.append(q_dict[line['q_type']])



word2vec = KeyedVectors.load_word2vec_format('data/sogou_vectors.bin', binary=True)

embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype = "float32")
embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

MAX_TOKENS = word2vec.syn0.shape[0]
embedding_dim = word2vec.syn0.shape[1]
hidden_dim_1 = 200
hidden_dim_2 = 100
NUM_CLASSES = 9
MAX_SEQUENCE_LENGTH = 50
VALIDATION_SPLIT = 0.1

document = Input(shape = (None, ), dtype = "int32")
left_context = Input(shape = (None, ), dtype = "int32")
right_context = Input(shape = (None, ), dtype = "int32")

embedder = Embedding(MAX_TOKENS + 1, embedding_dim, weights = [embeddings], trainable = False)
doc_embedding = embedder(document)
l_embedding = embedder(left_context)
r_embedding = embedder(right_context)

forward = LSTM(hidden_dim_1, return_sequences = True)(l_embedding)
backward = LSTM(hidden_dim_1, return_sequences = True, go_backwards = True)(r_embedding) 
together = concatenate([forward, doc_embedding, backward], axis = 2) 

semantic = TimeDistributed(Dense(hidden_dim_2, activation = "tanh"))(together)

pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (hidden_dim_2, ))(semantic) 

output = Dense(NUM_CLASSES, input_dim = hidden_dim_2, activation = "softmax")(pool_rnn) 

model = Model(inputs = [document, left_context, right_context], outputs = output)
model.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics = ["accuracy"])


doc_as_array = []
left_context_as_array = []
right_context_as_array = []


for text in q_zh:
    text = text.strip().lower().translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    tokens = jieba.cut(text, cut_all=False)
    tokens = [word2vec.vocab[token].index if token in word2vec.vocab else MAX_TOKENS for token in tokens]

    doc_as_array.append(np.array([tokens])[0])
   
    left_context_as_array.append(np.array([[MAX_TOKENS] + tokens[:-1]])[0])

    right_context_as_array.append(np.array([tokens[1:] + [MAX_TOKENS]])[0])



doc_as_array = pad_sequences(doc_as_array, maxlen=MAX_SEQUENCE_LENGTH)
left_context_as_array = pad_sequences(left_context_as_array, maxlen=MAX_SEQUENCE_LENGTH)
right_context_as_array = pad_sequences(right_context_as_array, maxlen=MAX_SEQUENCE_LENGTH)
target = to_categorical(q_type, num_classes=NUM_CLASSES)


perm = np.random.permutation(len(q_zh))
idx_train = perm[:int(len(q_zh)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(q_zh)*(1-VALIDATION_SPLIT)):]

doc_train = doc_as_array[idx_train]
left_train = left_context_as_array[idx_train]
right_train = right_context_as_array[idx_train]
target_train = target[idx_train]

doc_val = doc_as_array[idx_val]
left_val = left_context_as_array[idx_val]
right_val = right_context_as_array[idx_val]
target_val = target[idx_val]

timestr = time.strftime("%Y%m%d-%H%M%S")
STAMP = 'lstm_' + timestr

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([doc_train, left_train, right_train], target_train, validation_data=([doc_val, left_val, right_val], target_val), epochs = 200, batch_size=2048, shuffle=True, callbacks=[early_stopping, model_checkpoint])


model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])
