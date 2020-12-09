
def config_RNNEncDec():
    conf = {
    'vocab_size':10000, # maximum vocabulary size
    'max_sent_len':50, # maximum sequence length

# Model Arguments
    'attention': False,
    'emb_dim':120, # size of word embeddings
    'c_emb_dim':15, # size of char embeddings
    'n_hidden':1000, # number of hidden units per layer
    'n_layers':1, # number of layers
    'noise_radius':0.2, # stdev of noise for autoencoder (regularizer)
    'noise_anneal':0.995, # anneal noise_radius exponentially by this every 100 iterations
    'temp':1.0, # softmax temperature (lower --> more discrete)
    'dropout':0.5, # dropout applied to layers (0 = no dropout)
    'teach_force':0.5, # probability to use teach force

# Training Arguments
    'batch_size':100,
    'epochs':30, # maximum number of epochs
    'min_epochs':2, # minimum number of epochs to train for

    'lr':1e-4, # autoencoder learning rate
    'adam_epsilon':1e-8,
    'warmup_steps': 4000, #
    'clip':1.0,  # gradient clipping, max norm
    }
    return conf


def config_Transformer_EncoderDecoder():
    conf = {
    'vocab_size':10000, # maximum vocabulary size
    'max_sent_len':50, # maximum sequence length

# Model Arguments
    'emb_dim':120, # size of word embeddings
    'temp':1.0, # softmax temperature (lower --> more discrete)
    'dropout':0.5, # dropout applied to layers (0 = no dropout)
    'head':8, #multihead nums
    'd_ff':2048,
    'drop_out':0.1,
    'N_layers':3 , # sublayer nums
    
# Training Arguments
    'batch_size':200,
    'epochs':30, # maximum number of epochs
    'min_epochs':2, # minimum number of epochs to train for

    'lr':1e-4, # autoencoder learning rate
    'adam_epsilon':1e-8,
    'warmup_steps': 4000, #
    'clip':1.0,  # gradient clipping, max norm
    }
    return conf