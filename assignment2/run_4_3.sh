#!/bin/sh

## RNN
############################################################################
# num_layers
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=1 --dp_keep_prob=0.35 --save_best --save_dir=4_3_
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=3 --dp_keep_prob=0.35 --save_best --save_dir=4_3_

# keep prob
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.5 --save_best --save_dir=4_3_

# sequence length
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=50 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=4_3_
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=20 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=4_3_

# batch size
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=40 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=4_3_

# hidden size
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=2000 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=4_3_
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1000 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=4_3_

## GRU
############################################################################
# num_layers
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=1 --dp_keep_prob=0.35 --save_best --save_dir=4_3_
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=4_3_

# keep prob
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.5 --save_best --save_dir=4_3_

# sequence length
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=50 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=4_3_
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=20 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=4_3_

# batch size
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=50 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=4_3_

# hidden size
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=2000 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=4_3_
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1000 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=4_3_

## TRANSFORMER
############################################################################
