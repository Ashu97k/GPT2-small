# GPT2-small
Implementation and Optimization of GPT-2 Model, For Task1 and Task2 (Part1) running code is avaible for rest of the task the understanding from several references has been mentioned in this file.

# Task 1 | GPT-2 Model & Checkpoints 

the model is prepared from the dataset tiny shakespeare dataset (source: https://github.com/karpathy/nanoGPT and https://www.youtube.com/watch?v=kCc8FmEb1nY)

!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Task 2 | Transformer Architectural Changes
# Rotary Positional Embedding
Rotary embedding is used in forward function under Class Head in the below code

Findings:There is some improvement during training

References:

implementation in https://github.com/lucidrains/rotary-embedding-torch
su2021roformer, title - RoFormer: Enhanced Transformer with Rotary Position Embedding

# Group Query Attention

1. As mentioned in paper Ainslie et. al. GQA: Training Generalized Multi-Query Transformer, the Multi-head attention has H query, key, and value heads. Multi-query attention shares single key and value heads across all query heads. Grouped-query attention instead shares single key and value heads for each group of query heads, interpolating between multi-head and multi-query attention.
  ![image](https://github.com/Ashu97k/GPT2-small/assets/152495514/7e425e6b-6185-4753-b357-7bb8b71f3226)


3. Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head. GQA-G refers to grouped-query with G groups, so we need to denfine G value first.
For this we need to make changes in this part

```
class MultiHeadAttention(nn.Module):
""" multiple heads of self-attention are running in parallel """
  def __init__(self, num_heads, head_size):
      super().__init__()
      self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
      self.proj = nn.Linear(n_embd, n_embd)
      self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
      out = torch.cat([h(x) for h in self.heads], dim=-1)
      out = self.dropout(self.proj(out))
      return out
```
In this case the head size is same for queries, key and value so we need to define n_grouped and n_query_each_group as shown below
```
class GroupedQueryAttention(nn.Module):
  def __init__(self, head_size, n_embd, n_grouped: int = 4, n_query_each_group:int=2) -> None:
      super().__init__(head_size, n_embd)
      self.grouped = nn.ModuleList([Head(head_size, n_embd, n_query=n_query_each_group) for _ in range(n_grouped)])
      self.proj = nn.Linear(in_features=n_embd * n_grouped, out_features=n_embd, bias=False)
      self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
      out = torch.cat([h(x) for head in self.grouped], dim=1)
      out = self.dropout(self.proj(out))
      return out
````

In the original implementation from the reference the MQA need to be required along with GQA. The findings from paper is that the speed will be optimized.

# Sliding Window Attention

1. As mentioned in paper Beltagy et. al. Longformer: The Long-Document Transformer, is used for processing long documents and that makes it easy to perform a wide range of document-level NLP tasks without chunking/shortening the long input. It employs an attention pattern that combines local and global information while also scaling linearly with the sequence length.

2. The longformer code is understood from this repository https://github.com/allenai/longformer/tree/master/longformer
Below is a sample code:
```
import torch
from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer

config = LongformerConfig.from_pretrained('longformer-base-4096/')
# choose the attention mode 'n2', 'tvm' or 'sliding_chunks'
# 'n2': for regular n2 attantion
# 'tvm': a custom CUDA kernel implementation of our sliding window attention
# 'sliding_chunks': a PyTorch implementation of our sliding window attention
config.attention_mode = 'sliding_chunks'
model = Longformer.from_pretrained('longformer-base-4096/', config=config)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer.model_max_length = model.config.max_position_embeddings

SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document

input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

# Attention mask values -- 0: no attention, 1: local attention, 2: global attention
attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
attention_mask[:, [1, 4, 21,]] =  2  # Set global attention based on the task. For example,
                                     # classification: the <s> token
                                     # QA: question tokens

# padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
input_ids, attention_mask = pad_to_window_size(
        input_ids, attention_mask, config.attention_window[0], tokenizer.pad_token_id)

output = model(input_ids, attention_mask=attention_mask)[0]

```

# Task 3: Training Loop Implementation

# Distributed Data Parallel (DDP) 
PyTorch’s built-in features, such as DataParallel (DP) and DistributedDataParallel (DDP) offer accelerated training capabilities. It transparently performs distributed data parallel training. As an example that uses a torch.nn.Linear as the local model, it is wrapped with DDP, and then runs one forward pass, one backward pass, and an optimizer step on the DDP model. After that, parameters on the local model will be updated, and all models on different processes should be exactly the same. Library that can be import is from torch.nn.parallel import DistributedDataParallel as DDP.

source: https://pytorch.org/docs/master/notes/ddp.html
```
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
```

# Fully Sharded Data Parallel (FSDP)
This can be used to accelerate training huge models on larger batch sizes. It is a way of training models with parallel processing, similar to traditional data-parallel methods. However, unlike the usual approach that keeps separate copies of a model's parameters, gradients, and optimizer states for each GPU, FSDP divides and shares these states among the parallel workers. Additionally, it provides the option to move the divided model parameters to CPUs if needed.

Normally, FSDP wraps model layers in a nested manner. This means that only the layers within a specific FSDP instance have to bring all the parameters to a single device during forward or backward computations. Once the computation is done, the gathered parameters are released right away. This freed-up memory is then available for the next layer's computation. This process helps save peak GPU memory, allowing for the possibility of training with a larger model size or a larger batch size.

Using FSDP in pytorch - There are two ways to wrap a model with PyTorch FSDP. Auto wrapping serves as a seamless replacement for DDP, while manual wrapping requires only minor adjustments to the model definition code, offering the flexibility to experiment with intricate sharding strategies.

Source:https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/

```
#Auto wrapping

  from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload,
  )
  from torch.distributed.fsdp.wrap import (
    default_auto_wrap_policy,
  )
  import torch.nn as nn

  class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 4)
        self.layer2 = nn.Linear(4, 16)
        self.layer3 = nn.Linear(16, 4)

  model = DistributedDataParallel(model())
  fsdp_model = FullyShardedDataParallel(
    model(),
    fsdp_auto_wrap_policy=default_auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=True),
  )
```
