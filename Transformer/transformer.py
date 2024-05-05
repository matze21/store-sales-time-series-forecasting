import tensorflow as tf
tf.random.set_seed(42)

from . import encoder, decoder


class Transformer(tf.keras.Model):
  def __init__(self, *, enc_dec_num_layers, input_seq_dim_enc, input_seq_dim_dec, num_heads, fully_connected_size,
               input_sequence_len, output_sequence_len, dropout_rate=0.1):
    super().__init__(name='transformerBase')
    self.encoder = Encoder(num_layers=enc_dec_num_layers, d_model=input_seq_dim_enc,
                           num_heads=num_heads, dff=fully_connected_size,
                           sequence_length=input_sequence_len,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=enc_dec_num_layers, d_model=input_seq_dim_dec,
                           num_heads=num_heads, dff=fully_connected_size,
                           sequence_length=output_sequence_len,
                           dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(output_sequence_len, name='finallayer')

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    
    context, x  = inputs
    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)


    # needed if out inputs are not always the same length and the embedding only works for a part
    #try:
    #  # Drop the keras mask, so it doesn't scale the losses/metrics.
    #  # b/250038731
    #  del logits._keras_mask
    #except AttributeError:
    #  pass

    # Return the final output and the attention weights.
    return logits