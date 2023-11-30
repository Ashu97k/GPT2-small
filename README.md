# GPT2-small
Implementation and Optimization of GPT-2 Model

For Task 1, the model is prepared from the dataset tiny shakespeare dataset (source: https://github.com/karpathy/nanoGPT)

!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Task 2 | Transformer Architectural Changes
# Rotary Positional Embedding
Rotary embedding is used in forward function under Class Head in the below code

Findings:There is some improvement during training

References:

implementation in https://github.com/lucidrains/rotary-embedding-torch
su2021roformer, title - RoFormer: Enhanced Transformer with Rotary Position Embedding

# Group Query Attention

As mentioned in paper Ainslie et. al. GQA: Training Generalized Multi-Query Transformer, the Multi-head attention has H query, key, and value heads. Multi-query attention shares single key and value heads across all query heads. Grouped-query attention instead shares single key and value heads for each group of query heads, interpolating between multi-head and multi-query attention.
Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head. GQA-G refers to grouped-query with G groups, so we need to denfine G value first.
For this we need to make changes in this part



# Sliding Window Attention

As mentioned in paper Beltagy et. al. Longformer: The Long-Document Transformer, is used for processing long documents and that makes it easy to perform a wide range of document-level NLP tasks without chunking/shortening the long input. It employs an attention pattern that combines local and global information while also scaling linearly with the sequence length.
The longformer code is understood from this repository https://github.com/allenai/longformer/tree/master/longformer
Below is a sample code:
