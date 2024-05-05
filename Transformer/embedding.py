import tensorflow as tf
tf.random.set_seed(42)
import numpy as np
np.random.seed(42)

# 2d positional encoding (feature & position)
def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

def positional_encoding_2d(seq_length, num_features, wave_length=10000):
    # Create an array for the positions
    positions = np.arange(seq_length)[:, np.newaxis]
    # Create an array for the feature indices
    features = np.arange(num_features)[np.newaxis, :]
    # Calculate the divisors for the positional encoding
    divisors = np.exp(-2 * np.pi * features / wave_length)
    # Calculate the positional encoding
    pos_enc = positions * divisors
    # Return the positional encoding
    return tf.cast(pos_enc, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, sequence_length, d_model):
    super().__init__()
    self.d_model = d_model #d_model or model depth = dimensionality of encoding vector
    # no need for an embedding layer since my input is a time series and not a word token
    #self.embedding = tf.keras.layers.Embedding(sequence_length, d_model, mask_zero=True) 
    
    #self.pos_encoding = positional_encoding(length=sequence_length, depth=d_model)
    self.pos_encoding = positional_encoding_2d(sequence_length, d_model)

    #TODO implement an encoding for the store and family
    #self.store_fam_encoding


  def call(self, x):
    #print('encoder x', x.shape)
    length = tf.shape(x)[1]

    #x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

    x = x + self.pos_encoding#[tf.newaxis, :length, :]
    return x