import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).
    
    inputs: 
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1


class RNNLayer(nn.Module):
    """
    Defines a single RNN cell that is composed inside
    the RNN class
    """
    def __init__(self, x_size, hidden_size):
        """
        :param x_size:      x input size
        :param hidden_size: Hidden state input and output size
        """
        super(RNNLayer, self).__init__()
        # For efficiency weight vectors concatenated
        self.W = nn.Linear(x_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.hidden_size = hidden_size

    def forward(self, x, h):
        """
        :param x:   x input
        :param h:   Previous h hidden state h_{t-1}
        :return:    Hidden state output of cell
        """
        return self.tanh(self.W(torch.cat((x, h), 1)))

    def init_weights(self):
        """
        Initializes all weights to [-k, k] where
        k = 1/sqrt(hidden_size)
        """
        k = 1. / math.sqrt(self.hidden_size)
        torch.nn.init.uniform_(self.W.weight, -k, k)
        torch.nn.init.uniform_(self.W.bias, -k, k)


class RNNBase(nn.Module):

  def __init__(self, layer_ctor, emb_size, hidden_size, seq_len, batch_size,
               vocab_size, num_layers, dp_keep_prob, track_state_history=False):
    """
    :param layer_ctor:  Number of units in the input embeddings
    :param emb_size:    Number of hidden units per layer
    :param hidden_size: Length of the input sequences
    :param seq_len:     Length of the input sequences
    :param batch_size:  Batch size of data
    :param vocab_size:  Number of tokens in the vocabulary
    :param num_layers:  Number of hidden layers in network
    :param dp_keep_prob:The probability of *not* dropping out units
    :param track_state_history: If to track all state history (for 5.2)
    """
    super(RNNBase, self).__init__()

    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.dp_keep_prob = dp_keep_prob

    self.rnn_layers = nn.ModuleList()
    self.dropout_layers = nn.ModuleList()

    self.rnn_layers.extend([layer_ctor(emb_size if i == 0 else hidden_size, hidden_size)
                            for i in range(num_layers)])
    self.dropout_layers.extend([nn.Dropout(1-dp_keep_prob)
                                for i in range(num_layers)])
    self.output_layer = nn.Linear(hidden_size, vocab_size)

    self.embedding_layer = nn.Embedding(vocab_size, emb_size)
    self.embedding_dropout = nn.Dropout(1-dp_keep_prob)
    self.track_state_history = track_state_history
    self.state_history = None
    self.init_weights()

  def init_weights(self):
    """
    Initializes embedding and output weights initialized to [-0.1, 0.1].
    Output bias initialized to 0s
    Recurrent layer initialized to [-k, k] where k = 1/sqrt(hidden_size)
    """
    torch.nn.init.uniform_(self.embedding_layer.weight, -0.1, 0.1)
    torch.nn.init.uniform_(self.output_layer.weight, -0.1, 0.1)
    torch.nn.init.zeros_(self.output_layer.bias)
    for rnn_layer in self.rnn_layers:
        rnn_layer.init_weights()

  def init_hidden(self):
    """
    Creates the initial hidden state
    """
    return torch.zeros([self.num_layers, self.batch_size, self.hidden_size])

  def forward(self, inputs, hidden):
    """
    :param inputs:  A mini-batch of input sequences,
                    composed of int ids representing vocabulary
    :param hidden:  Initial hidden states for every layer of the stacked RNN.
                    shape: (num_layers, batch_size, hidden_size)
    :return:        Tuple of output logits and final hidden state.
                    Shape (seq_len, batch_size, vocab_size)
                    and (num_layers, batch_size, hidden_size) respectively
    """
    logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size], device=inputs.device)

    # Used for 5.2 to track all hidden states for gradients
    if self.track_state_history:
        self.state_history = [[] for _ in range(self.num_layers)]

    embedding_output = self.embedding_layer(inputs)

    # For each time-step compute t'th output by looping upwards in layers.
    # Hidden state is stored for next t+1 chain.
    # Embedding layer and recurrent cells are followed by dropout
    for t in range(self.seq_len):
        x = self.embedding_dropout(embedding_output[t])
        h_t = []
        for l in range(self.num_layers):
            h_out = self.rnn_layers[l](x, hidden[l])
            x = self.dropout_layers[l](h_out)
            h_t.append(h_out)

            # Used for 5.2 to track all hidden states for gradients
            if self.track_state_history:
                self.state_history[l].append(h_out)

        # Form new hidden state tensor for next time-step
        hidden = torch.stack(h_t)
        logits[t] = self.output_layer(x)

    return logits, hidden

  def generate(self, input, hidden, generated_seq_len):
    """
    :param input:   A mini-batch of input tokens
                    shape: (batch_size)
    :param hidden:  The initial hidden states for every layer of the stacked RNN
                    shape: (num_layers, batch_size, hidden_size)
    :param generated_seq_len:
                    The length of the sequence to generate
                    shape: (num_layers, batch_size, hidden_size)
    :return:        Sampled sequences of tokens
                    shape: (generated_seq_len, batch_size)
    """
    hidden_states = hidden.clone()
    current_word = input
    samples = torch.zeros((generated_seq_len, input.shape[0]), device=input.device)

    for t in range(generated_seq_len):
        x = self.embedding_dropout(self.embedding_layer(current_word))
        for l in range(self.num_layers):
            hidden_states[l] = self.rnn_layers[l](x, hidden_states[l])
            x = self.dropout_layers[l](hidden_states[l])

        # Predicted word fed back through network as next current_word
        current_word = torch.distributions.Categorical(
            logits=self.output_layer(x)).sample()
        samples[t] = current_word

    return samples


class RNN(RNNBase):
  """
  Implements an RNN recurrent network. Composes RNNLayer cells.
  """
  def __init__(self, emb_size, hidden_size, seq_len,
               batch_size, vocab_size, num_layers, dp_keep_prob):
    """
    :param emb_size:    The number of units in the input embeddings
    :param hidden_size: The number of hidden units per layer
    :param seq_len:     The length of the input sequences
    :param batch_size:  Batch size of data
    :param vocab_size:  The number of tokens in the vocabulary
    :param num_layers:  The depth of the stack (number of hidden layers)
    :param dp_keep_prob:  The probability of *not* dropping out units in the
                          non-recurrent connections.
    """
    super(RNN, self).__init__(RNNLayer, emb_size, hidden_size, seq_len,
                              batch_size, vocab_size, num_layers, dp_keep_prob)


# Problem 2
class GRULayer(nn.Module):
    """
    Implements a GRU cell composed in the GRU class
    """
    def __init__(self, x_size, hidden_size):
        """
        :param x_size:      x input size
        :param hidden_size: Hidden state input and output size
        """
        super(GRULayer, self).__init__()
        # For efficiency weight vectors concatenated
        self.r_linear = nn.Linear(x_size + hidden_size, hidden_size)
        self.z_linear = nn.Linear(x_size + hidden_size, hidden_size)
        self.h_linear = nn.Linear(x_size + hidden_size, hidden_size)
        self.h_tanh = nn.Tanh()
        self.r_sigmoid = nn.Sigmoid()
        self.z_sigmoid = nn.Sigmoid()
        self.hidden_size = hidden_size

    def forward(self, x, h_prev):
        """
        Executes forward pass through RNN cell.
        :param x:       x input
        :param h_prev:  Previous h hidden state h_{t-1}
        :return:        Hidden state output of cell
        """
        combined_input = torch.cat((x, h_prev), 1)
        z = self.z_sigmoid(self.z_linear(combined_input))
        r = self.r_sigmoid(self.r_linear(combined_input))
        h_candidate = self.h_tanh(self.h_linear(torch.cat((x, r*h_prev), 1)))
        return (1-z)*h_prev + z*h_candidate

    def init_weights(self):
        """
        Initializes all weights to [-k, k] where
        k = 1/sqrt(hidden_size)
        """
        k = 1. / math.sqrt(self.hidden_size)
        torch.nn.init.uniform_(self.r_linear.weight, -k, k)
        torch.nn.init.uniform_(self.r_linear.bias, -k, k)
        torch.nn.init.uniform_(self.z_linear.weight, -k, k)
        torch.nn.init.uniform_(self.z_linear.bias, -k, k)
        torch.nn.init.uniform_(self.h_linear.weight, -k, k)
        torch.nn.init.uniform_(self.h_linear.bias, -k, k)


class GRU(RNNBase):
  """
  Implements a GRU recurrent network. Composes GRULayer cells.
  """
  def __init__(self, emb_size, hidden_size, seq_len, batch_size,
               vocab_size, num_layers, dp_keep_prob):
    super(GRU, self).__init__(GRULayer, emb_size, hidden_size, seq_len,
                              batch_size, vocab_size, num_layers, dp_keep_prob)


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


#----------------------------------------------------------------------------------

class SingleHeadAttention(nn.Module):
    """
    Implements a single attention head class composed in
    MultiHeadedAttention. Each head computes an a_i / h_i result
    """
    EPSILON = 1e9

    def __init__(self, n_units, d_k, dropout_rate):
        """
        n_units:    Number of units in the attention head
        d_k:        Key output size
        dropout_rate: Rate to drop units
        """
        super(SingleHeadAttention, self).__init__()
        self.n_units = n_units
        self.d_k = d_k
        self.q_linear = nn.Linear(self.n_units, self.d_k)
        self.k_linear = nn.Linear(self.n_units, self.d_k)
        self.v_linear = nn.Linear(self.n_units, self.d_k)
        self.dropout = nn.Dropout(dropout_rate)

    def init_weights(self):
        """
        Initializes all weights to [-k, k] where
        k = 1/sqrt(n_units)
        """
        k = 1. / math.sqrt(self.n_units)
        nn.init.uniform_(self.q_linear.weight, -k, k)
        nn.init.uniform_(self.q_linear.bias, -k, k)
        nn.init.uniform_(self.k_linear.weight, -k, k)
        nn.init.uniform_(self.k_linear.bias, -k, k)
        nn.init.uniform_(self.v_linear.weight, -k, k)
        nn.init.uniform_(self.v_linear.bias, -k, k)

    def forward(self, query, key, value, mask=None):
        """
        Computes a single attention a_i / h_i result
        :param query:   Query matrix Q (batch_size, seq_len, n_units)
        :param key:     Key matrix K (batch_size, seq_len, n_units)
        :param value:   Value matrix V (batch_size, seq_len, n_units)
        :param mask:    Mask specifying whether to attend each element
                        (batch_size, seq_len, seq_len)
        """
        # Computes intermediate x value before compute a_i
        q_out = self.q_linear(query)
        k_out = self.k_linear(key)
        v_out = self.v_linear(value)
        x = torch.matmul(q_out, k_out.transpose(1, 2))
        x = torch.div(x, math.sqrt(self.d_k))

        # Apply mask
        if mask is not None:
            x = x * mask - SingleHeadAttention.EPSILON * (1 - mask)

        # Output attention head value
        a = F.softmax(x, dim=-1)
        a = self.dropout(a)
        return torch.matmul(a, v_out)


class MultiHeadedAttention(nn.Module):
    """
    Implements the multi-head scaled dot-product attention
    component of a transformer. Composes SingleHeadAttention.
    """
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        :param n_heads: the number of attention heads
        :param n_units: the number of output units
        :param dropout: probability of dropping units
        """
        super(MultiHeadedAttention, self).__init__()
        # Size of the keys, values, and queries (self.d_k)
        # is output units divided by the number of heads.
        self.d_k = n_units // n_heads
        assert n_units % n_heads == 0

        self.n_heads = n_heads
        self.n_units = n_units

        self.out_linear = nn.Linear(n_units, n_units)
        self.attention_heads = clones(SingleHeadAttention(n_units, self.d_k,
                                                          dropout), n_heads)
        self.init_weights()

    def init_weights(self):
        """
        Initializes all weights to [-k, k] where k = 1/sqrt(hidden_size)
        """
        k = 1. / math.sqrt(self.n_units)
        nn.init.uniform_(self.out_linear.weight, -k, k)
        nn.init.uniform_(self.out_linear.bias, -k, k)
        for attention_head in self.attention_heads:
            attention_head.init_weights()

    def forward(self, query, key, value, mask=None):
        """
        Compute multi-head scaled dot product attention
        :param query:   Query matrix Q (batch_size, seq_len, n_units)
        :param key:     Key matrix K (batch_size, seq_len, n_units)
        :param value:   Value matrix V (batch_size, seq_len, n_units)
        :param mask:    Mask specifying whether to attend each element
                        (batch_size, seq_len, seq_len)
        """

        # Mask preemptively converted to float for purposes of
        # tensor multiplication x * s - 1e9*(1-s)
        if mask is not None:
            mask = mask.float()

        # Compute each a_i output (see SingleHeadAttention),
        # concatenate all together and put through final linear output
        h_out = torch.cat([atn(query, key, value, mask)
                           for atn in self.attention_heads], dim=-1)
        return self.out_linear(h_out)


#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

