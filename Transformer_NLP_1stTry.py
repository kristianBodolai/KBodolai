'''Trying to build a transformer for natural language processing. Results where nos that good, needs fine tuning
    for this I have based myself in this post: http://www.peterbloem.nl/blog/transformers
    and the TensorFlow website tutorials. Training with a corpus of blog posts.'''

# %% markdown
# # **TEXT GENERATION TRANSFORMER**
# %% markdown
# # Imports
# %% codecell
!pip install -U sacremoses
# %% codecell
import tensorflow as tf
import tensorflow_datasets as tfds
import random
import numpy as np
import pandas as pd
from sacremoses import MosesPunctNormalizer
import tensorflow.keras as kr
import time
import os


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# %% codecell
PATH = "/media/Storage/text_corpus/blogtext.csv"
N_ENTRIES = 60000 #Has to be big enough to contain the train, test and validation data.

seq_length = 120
n_train = int(13e6//seq_length*seq_length)
n_test = int(5e5//seq_length*seq_length)
n_val = int(5e5//seq_length*seq_length)

n_args = [0, n_train, n_test, n_val]
# %% markdown
# # Classes
# %% codecell
class SelfAttentionWide(kr.layers.Layer):
  def __init__(self, k, heads = 8):
    super(SelfAttentionWide, self).__init__()

    self.k, self.heads = k, heads

    self.tokeys = kr.layers.Dense(k*heads)
    self.toqueries = kr.layers.Dense(k*heads)
    self.tovalues = kr.layers.Dense(k*heads)

    self.unifyheads = kr.layers.Dense(k)

  def call(self, x):
    b, t, k = x.shape
    h = self.heads

    keys = tf.reshape(self.tokeys(x), (b, t, h, k))
    queries = tf.reshape(self.toqueries(x), (b, t, h, k))
    values = tf.reshape(self.tovalues(x), (b, t, h, k))

    keys = tf.reshape(tf.transpose(keys, perm = [0, 2, 1, 3]), (b*h, t, k))
    queries = tf.reshape(tf.transpose(queries, perm = [0, 2, 1, 3]), (b*h, t, k))
    values = tf.reshape(tf.transpose(values, perm = [0, 2, 1, 3]), (b*h, t, k))

    queries = queries / (k ** (1/4))
    keys    = keys / (k ** (1/4))

    dot = tf.matmul(queries, tf.transpose(keys, perm = [0, 2, 1]))

    mask = 1 - tf.linalg.band_part(tf.ones((t, t)), -1, 0)
    dot += mask*(-1e9)


    dot = tf.nn.softmax(dot, axis=2)
    out = tf.reshape(tf.matmul(dot, values), (b, h, t, k))

    out = tf.reshape(tf.transpose(out, perm = [0, 2, 1, 3]), (b, t, h * k))
    return self.unifyheads(out)

class SelfAttentionNarrow(kr.layers.Layer):
  def __init__(self, k, heads = 8):
    super().__init__()

    self.k, self.heads = k, heads


class TransformerLayer(kr.layers.Layer):
  def __init__(self, k, heads = 8, rate=0.0):
    super(TransformerLayer, self).__init__()

    self.attention = SelfAttentionWide(k, heads=heads)

    self.norm1 = tf.keras.layers.LayerNormalization()
    self.norm2 = tf.keras.layers.LayerNormalization()

    self.ff = tf.keras.Sequential([tf.keras.layers.Dense(4*k, activation = 'relu'),
                                   tf.keras.layers.Dense(k)])

    self.dropout1 = tf.keras.layers.Dropout(rate)

  def call(self, x, training):
    attended = self.attention(x)
    x = self.norm1(attended + x)

    fedforward = self.ff(x)
    x =  self.norm2(fedforward + x)
    return self.dropout1(x, training)

class GenTransformer(kr.Model):
  def __init__(self, k, heads, depth, seq_length, num_tokens):
    super(GenTransformer, self).__init__()

    self.depth = depth
    self.num_tokens = num_tokens
    self.token_embedding = tf.keras.layers.Embedding(num_tokens, k)
    self.pos_embedding = tf.keras.layers.Embedding(seq_length, k)

    self.trans_layers = [TransformerLayer(k = k, heads=heads) for _ in range(depth)]

    self.toprobs = tf.keras.layers.Dense(num_tokens)

  def call(self, x, training):
    tokens = self.token_embedding(x)
    b, t, k = tokens.shape

    # generate position embeddings
    positions = tf.range(t)
    positions = self.pos_embedding(positions)[None, :, :]

    x = tokens + positions
    for i in range(self.depth):
      x = self.trans_layers[i](x, training)

    # Average-pool over the t dimension and project to class
    # probabilities
    x = tf.reshape(self.toprobs(tf.reshape(x, (b*t, k))),(b, t, self.num_tokens))
    return x

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, k, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.k = k
    self.k = tf.cast(self.k, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.k) * tf.math.minimum(arg1, arg2)
# %% markdown
# # Pipeline
# %% codecell
mpn = MosesPunctNormalizer()

file = pd.read_csv(PATH)
X_list = [str.encode(mpn.normalize(entry)) for entry in file['text'][:N_ENTRIES]]
random.shuffle(X_list)

#Train the subword tokenizer
tokenizer= tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (entry for entry in X_list[:300]), target_vocab_size=2**13)

#Tokenize data
X = tokenizer.encode(b'\n\n'.join(X_list))

#Prepare raw datasets
x_train, x_test, x_val = [tf.reshape(X[i:j], [-1, seq_length]) for i, j in zip(np.cumsum(n_args), np.cumsum(n_args)[1:])]

#Prepare batches
ds_train = tf.data.Dataset.from_tensor_slices(x_train).batch(32)
ds_test = tf.data.Dataset.from_tensor_slices(x_test).batch(64)
ds_val = tf.data.Dataset.from_tensor_slices(x_val).batch(64)
# %% markdown
# # Train
# %% codecell
depth = 7
k = 250
heads = 8
EPOCHS = 10
num_tokens = tokenizer.vocab_size + 2
print(num_tokens)
# %% codecell
learning_rate = CustomSchedule(k)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
#optimizer = tf.keras.optimizers.Adam()
# %% codecell
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
# %% codecell
transformer = GenTransformer(k, heads, depth, seq_length, num_tokens)
# %% codecell
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
# %% codecell
@tf.function()
def train_step(sec):
  inp = sec[:, :sec.shape[1]-1]
  tar = sec[:, 1:]

  with tf.GradientTape() as tape:
    predictions = transformer(inp, True)

    loss = loss_function(tar, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(tar, predictions)
# %% codecell
def generate(sec, length):
  inp = tf.expand_dims(tokenizer.encode(sec), 0)
  for s in sec:
    print(s, end = '', flush=True)
  for i in range(length):
    predictions = transformer(inp, False)
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    if inp.shape[1]>=seq_length:
      inp = tf.concat([inp[:, 1:], predicted_id], axis=1)
    else:
      inp = tf.concat([inp, predicted_id], axis=1)
    print(tokenizer.decode(predicted_id[:, 0]), end='', flush=True)
# %% codecell
phrase = tokenizer.decode(x_train[0])
generate(phrase, 100)
# %% codecell
for epoch in range(EPOCHS):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  for (batch, sec) in enumerate(ds_train):
    train_step(sec)

    if batch % 50 == 0:
      print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
          epoch + 1, batch, train_loss.result(), train_accuracy.result()))


  print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                train_loss.result(),
                                                train_accuracy.result()))

  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
# %% codecell
generate('The other day I was struck by a lightning and i found out that the meaning of life is ' , 50)

# %% codecell
transformer.summary()

tokenizer.save_to_file('tokenizer')
#ransformer.save('transformer30')
transformer.save_weights('transformer_w_and_b.h5')
