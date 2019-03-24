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

    def __init__(self, x_size, hidden_size):
        super(RNNLayer, self).__init__()
        self.W = nn.Linear(x_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.hidden_size = hidden_size

    def forward(self, x, h):
        return self.tanh(self.W(torch.cat((x, h), 1)))

    def init_weights(self):
        k = 1. / math.sqrt(self.hidden_size)
        torch.nn.init.uniform_(self.W.weight, -k, k)
        torch.nn.init.uniform_(self.W.bias, -k, k)


class RNNBase(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.

  def __init__(self, layer_ctor, emb_size, hidden_size, seq_len, batch_size,
               vocab_size, num_layers, dp_keep_prob, track_state_history=False):
    """
    emb_size:     The number of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(RNNBase, self).__init__()

    # TODO ========================
    # Initialization of the parameters of the recurrent and fc layers.
    # Your implementation should support any number of stacked hidden layers
    # (specified by num_layers), use an input embedding layer, and include fully
    # connected layers with dropout after each recurrent layer.
    # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding
    # modules, but not recurrent modules.
    #
    # To create a variable number of parameter tensors and/or nn.Modules
    # (for the stacked hidden layer), you may need to use nn.ModuleList or the
    # provided clones function (as opposed to a regular python list), in order
    # for Pytorch to recognize these parameters as belonging to this nn.Module
    # and compute their gradients automatically. You're not obligated to use the
    # provided clones function.
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.dp_keep_prob = dp_keep_prob

    self.rnn_layers = nn.ModuleList()
    self.dropout_layers = nn.ModuleList()

    self.rnn_layers.extend([layer_ctor(emb_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
    self.dropout_layers.extend([nn.Dropout(1-dp_keep_prob) for i in range(num_layers)])
    self.output_layer = nn.Linear(hidden_size, vocab_size)

    self.embedding_layer = nn.Embedding(vocab_size, emb_size)
    self.embedding_dropout = nn.Dropout(1-dp_keep_prob)
    self.track_state_history = track_state_history
    self.state_history = None
    self.init_weights()

  def init_weights(self):
    # TODO ========================
    # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
    # and output biases to 0 (in place). The embeddings should not use a bias vector.
    # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly
    # in the range [-k, k] where k is the square root of 1/hidden_size
    torch.nn.init.uniform_(self.embedding_layer.weight, -0.1, 0.1)
    torch.nn.init.uniform_(self.output_layer.weight, -0.1, 0.1)
    torch.nn.init.zeros_(self.output_layer.bias)
    for rnn_layer in self.rnn_layers:
        rnn_layer.init_weights()


  def init_hidden(self):
    # TODO ========================
    # initialize the hidden states to zero
    """
    This is used for the first mini-batch in an epoch, only.
    """
    # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
    return torch.zeros([self.num_layers, self.batch_size, self.hidden_size])

  def forward(self, inputs, hidden):
    # TODO ========================
    # Compute the forward pass, using nested python for loops.
    # The outer for loop should iterate over timesteps, and the
    # inner for loop should iterate over hidden layers of the stack.
    #
    # Within these for loops, use the parameter tensors and/or nn.modules you
    # created in __init__ to compute the recurrent updates according to the
    # equations provided in the .tex of the assignment.
    #
    # Note that those equations are for a single hidden-layer RNN, not a stacked
    # RNN. For a stacked RNN, the hidden states of the l-th layer are used as
    # inputs to to the {l+1}-st layer (taking the place of the input sequence).

    """
    Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that
                    represent the index of the current token(s) in the vocabulary.
                        shape: (seq_len, batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)

    Returns:
        - Logits for the softmax over output tokens at every time-step.
              **Do NOT apply softmax to the outputs!**
              Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does
              this computation implicitly.
                    shape: (seq_len, batch_size, vocab_size)
        - The final hidden states for every layer of the stacked RNN.
              These will be used as the initial hidden states for all the
              mini-batches in an epoch, except for the first, where the return
              value of self.init_hidden will be used.
              See the repackage_hiddens function in ptb-lm.py for more details,
              if you are curious.
                    shape: (num_layers, batch_size, hidden_size)
    """
    logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size], device=inputs.device)

    # Used for 5.2 to keep track of all hidden states for gradient calculations
    if self.track_state_history:
        self.state_history = [[] for _ in range(self.num_layers)]

    embedding_output = self.embedding_layer(inputs)

    for t in range(self.seq_len):
        x = self.embedding_dropout(embedding_output[t])
        h_t = []
        for l in range(self.num_layers):
            h_out = self.rnn_layers[l](x, hidden[l])
            x = self.dropout_layers[l](h_out)
            h_t.append(h_out)

            # Used for 5.2 to keep track of all hidden states for gradient calculations
            if self.track_state_history:
                self.state_history[l].append(h_out)

        hidden = torch.stack(h_t)
        logits[t] = self.output_layer(x)

    return logits, hidden

  def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
    # Compute the forward pass, as in the self.forward method (above).
    # You'll probably want to copy substantial portions of that code here.
    #
    # We "seed" the generation by providing the first inputs.
    # Subsequent inputs are generated by sampling from the output distribution,
    # as described in the tex (Problem 5.3)
    # Unlike for self.forward, you WILL need to apply the softmax activation
    # function here in order to compute the parameters of the categorical
    # distributions to be sampled from at each time-step.

    """
    Arguments:
        - input: A mini-batch of input tokens (NOT sequences!)
                        shape: (batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
        - generated_seq_len: The length of the sequence to generate.
                       Note that this can be different than the length used
                       for training (self.seq_len)
    Returns:
        - Sampled sequences of tokens
                    shape: (generated_seq_len, batch_size)
    """

    hidden_states = hidden.clone()
    current_word = input
    samples = torch.zeros((generated_seq_len, input.shape[0]), device=input.device)

    for t in range(generated_seq_len):
        x = self.embedding_dropout(self.embedding_layer(current_word))
        for l in range(self.num_layers):
            hidden_states[l] = self.rnn_layers[l](x, hidden_states[l])  # Forward pass
            x = self.dropout_layers[l](hidden_states[l])
        current_word = torch.distributions.Categorical(logits=self.output_layer(x)).sample()
        samples[t] = current_word

    return samples


class RNN(RNNBase): # Implement a stacked vanilla RNN with Tanh nonlinearities.

  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    """
    emb_size:     The number of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(RNN, self).__init__(RNNLayer, emb_size, hidden_size, seq_len, batch_size,
                              vocab_size, num_layers, dp_keep_prob)


# Problem 2
class GRULayer(nn.Module):
    def __init__(self, x_size, hidden_size):
        super(GRULayer, self).__init__()
        self.r_linear = nn.Linear(x_size + hidden_size, hidden_size)
        self.z_linear = nn.Linear(x_size + hidden_size, hidden_size)
        self.h_linear = nn.Linear(x_size + hidden_size, hidden_size)
        self.h_tanh = nn.Tanh()
        self.r_sigmoid = nn.Sigmoid()
        self.z_sigmoid = nn.Sigmoid()
        self.hidden_size = hidden_size

    def forward(self, x, h_prev):
        combined_input = torch.cat((x, h_prev), 1)
        z = self.z_sigmoid(self.z_linear(combined_input))
        r = self.r_sigmoid(self.r_linear(combined_input))
        h_candidate = self.h_tanh(self.h_linear(torch.cat((x, r*h_prev), 1)))
        return (1-z)*h_prev + z*h_candidate

    def init_weights(self):
        k = 1. / math.sqrt(self.hidden_size)
        torch.nn.init.uniform_(self.r_linear.weight, -k, k)
        torch.nn.init.uniform_(self.r_linear.bias, -k, k)
        torch.nn.init.uniform_(self.z_linear.weight, -k, k)
        torch.nn.init.uniform_(self.z_linear.bias, -k, k)
        torch.nn.init.uniform_(self.h_linear.weight, -k, k)
        torch.nn.init.uniform_(self.h_linear.bias, -k, k)


class GRU(RNNBase): # Implement a stacked GRU RNN
  """
  Follow the same instructions as for RNN (above), but use the equations for
  GRU, not Vanilla RNN.
  """
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
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

# TODO: implement this class
class SingleHeadAttention(nn.Module):

    EPSILON = 1e9

    def __init__(self, n_units, d_k, dropout_rate):
        super(SingleHeadAttention, self).__init__()
        self.n_units = n_units
        self.d_k = d_k
        self.q_linear = nn.Linear(self.n_units, self.d_k)
        self.k_linear = nn.Linear(self.n_units, self.d_k)
        self.v_linear = nn.Linear(self.n_units, self.d_k)
        self.dropout = nn.Dropout(dropout_rate)

    def init_weights(self):
        k = 1. / math.sqrt(self.n_units)
        nn.init.uniform_(self.q_linear.weight, -k, k)
        nn.init.uniform_(self.q_linear.bias, -k, k)
        nn.init.uniform_(self.k_linear.weight, -k, k)
        nn.init.uniform_(self.k_linear.bias, -k, k)
        nn.init.uniform_(self.v_linear.weight, -k, k)
        nn.init.uniform_(self.v_linear.bias, -k, k)

    def forward(self, query, key, value, mask=None):
        q_out = self.q_linear(query)
        k_out = self.k_linear(key)
        v_out = self.v_linear(value)
        x = torch.matmul(q_out, k_out.transpose(1, 2))
        x = torch.div(x, math.sqrt(self.d_k))

        if mask is not None:
            x = x * mask - SingleHeadAttention.EPSILON * (1 - mask)

        a = F.softmax(x, dim=-1)
        a = self.dropout(a)
        return torch.matmul(a, v_out)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_heads = n_heads
        self.n_units = n_units

        # TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear
        # and nn.Dropout
        # ETA: you can also use softmax
        # ETA: you can use the "clones" function we provide.
        # ETA: you can use masked_fill

        self.out_linear = nn.Linear(n_units, n_units)
        self.attention_heads = clones(SingleHeadAttention(n_units, self.d_k, dropout), n_heads)
        self.init_weights()

    def init_weights(self):
        k = 1. / math.sqrt(self.n_units)
        nn.init.uniform_(self.out_linear.weight, -k, k)
        nn.init.uniform_(self.out_linear.bias, -k, k)
        for attention_head in self.attention_heads:
            attention_head.init_weights()

    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value correspond to Q, K, and V in the latex, and
        # they all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.

        if mask is not None:
            mask = mask.float()

        h_out = torch.cat([atn(query, key, value, mask) for atn in self.attention_heads], dim=-1)
        return self.out_linear(h_out)


class MultiHeadedAttentionAlternative(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttentionAlternative, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_heads = n_heads
        self.n_units = n_units

        # TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear
        # and nn.Dropout
        # ETA: you can also use softmax
        # ETA: you can use the "clones" function we provide.
        # ETA: you can use masked_fill

        # Creating 4 identical linear layers (for Qo, Ko, Vo and final output layer)
        # with their own parameters.
        self.lin_query = nn.Linear(n_units,n_units)
        self.lin_key = nn.Linear(n_units,n_units)
        self.lin_value = nn.Linear(n_units,n_units)
        self.lin_out = nn.Linear(n_units,n_units)

        self.dropout = nn.Dropout(dropout)

        # Weights and biases of the linear layers initialization
        self.lin_query.weight.data.uniform_(-math.sqrt(1/n_units), math.sqrt(1/n_units))
        self.lin_query.bias.data.uniform_(-math.sqrt(1/n_units), math.sqrt(1/n_units))
        self.lin_key.weight.data.uniform_(-math.sqrt(1/n_units), math.sqrt(1/n_units))
        self.lin_key.bias.data.uniform_(-math.sqrt(1/n_units), math.sqrt(1/n_units))
        self.lin_value.weight.data.uniform_(-math.sqrt(1/n_units), math.sqrt(1/n_units))
        self.lin_value.bias.data.uniform_(-math.sqrt(1/n_units), math.sqrt(1/n_units))
        self.lin_out.weight.data.uniform_(-math.sqrt(1/n_units), math.sqrt(1/n_units))
        self.lin_out.bias.data.uniform_(-math.sqrt(1/n_units), math.sqrt(1/n_units))

    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value correspond to Q, K, and V in the latex, and
        # they all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.

        # Keep batch size
        nb_batch = query.size(0)

        # Compute linear system for query, key and value
        # Divide each components into d_k elements in parallel
        Qo = self.lin_query(query).view(nb_batch, -1, self.n_heads, self.d_k)
        Ko = self.lin_key(key).view(nb_batch, -1, self.n_heads, self.d_k)
        Vo = self.lin_value(value).view(nb_batch, -1, self.n_heads, self.d_k)

        # Scaled dot-product attention
        Ko = torch.transpose(Ko,1,2)
        Qo = torch.transpose(Qo,1,2)
        Vo = torch.transpose(Vo,1,2)

        A = torch.matmul(Qo,Ko.transpose(-2, -1))
        A = A/math.sqrt(self.d_k)
        A = A.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        A = nn.functional.softmax(A, dim=-1) # Softmax on the rightmost dimension

        # Dropout attention values
        A = self.dropout(A)

        # Apply attention on values
        H = torch.matmul(A, Vo)

        # Concatenate the attention heads
        H = H.transpose(1, 2).contiguous().view(nb_batch, -1, self.n_units)

        # Apply linear layer on attentions
        Ao = self.lin_out(H)

        return Ao # size: (batch_size, seq_len, self.n_units)


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
