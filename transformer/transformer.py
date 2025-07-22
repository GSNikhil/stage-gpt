import torch.nn as nn
from torch.nn import functional as F
import torch
import yaml

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MaskedMultiHeadAttention, self).__init__()
        self.num_heads = config['num_heads']
        self.d_per_head = config['d_model'] // self.num_heads

        self.Wq = nn.Linear(config['d_model'], config['d_model'])
        self.Wk = nn.Linear(config['d_model'], config['d_model'])
        self.Wv = nn.Linear(config['d_model'], config['d_model'])
        self.register_buffer(
            name="attention_mask", 
            tensor=torch.triu(torch.ones(config['context_length'], config['context_length']), diagonal=1),
            persistent=False    # Dont need a state
        )
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.projection = nn.Linear(config['d_model'], config['d_model'])
        
    def forward(self, x):
        # x is of shape (batch x num_tokens x d_model)
        batch, num_tokens, d_model = x.shape
        """
        queries = self.Wq(x)    # b x num_tokens x d_model
        keys = self.Wk(x)       # b x num_tokens x d_model
        values = self.Wv(x)     # b x num_tokens x d_model

        queries = queries.reshape(batch, num_tokens, self.num_heads, self.d_per_head)     # b x num_tokens x num_heads x self.d_per_head
        keys = keys.reshape(batch, num_tokens, self.num_heads, self.d_per_head)        # b x num_tokens x num_heads x self.d_per_head
        values = values.reshape(batch, num_tokens, self.num_heads, self.d_per_head)      # b x num_tokens x num_heads x self.d_per_head

        queries = queries.transpose(1, 2)   # b x num_heads x num_tokens x self.d_per_head
        keys = keys.transpose(1, 2)         # b x num_heads x num_tokens x self.d_per_head
        values = values.transpose(1, 2)     # b x num_heads x num_tokens x self.d_per_head
        """
        queries = self.Wq(x).reshape(batch, num_tokens, self.num_heads, self.d_per_head).transpose(1, 2)
        keys = self.Wk(x).reshape(batch, num_tokens, self.num_heads, self.d_per_head).transpose(1, 2)
        values = self.Wv(x).reshape(batch, num_tokens, self.num_heads, self.d_per_head).transpose(1, 2)

        attention_scores = queries @ keys.transpose(-1, -2)     # b x num_heads x num_tokens x num_tokens
        attention_scores = attention_scores / (self.d_per_head ** 0.5)
        # Now mask the attention scores and pass through dropout
        attention_scores.masked_fill_(self.attention_mask[:num_tokens, :num_tokens].bool(), -torch.inf) # this is causal attention
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)       

        # Get context vectors
        context_vector = attention_scores @ values      # b x num_heads x num_tokens x self.d_per_head
        context_vector = context_vector.transpose(1, 2) # b x num_tokens x num_heads x self.d_per_head

        context_vector = context_vector.reshape(batch, num_tokens, -1)
        return self.projection(context_vector)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(config['d_model'])
        self.attention = MaskedMultiHeadAttention(config)
        self.dropout = nn.Dropout(config['dropout_rate'])

        self.ln_2 = nn.LayerNorm(config['d_model'])
        self.ff = nn.Sequential(
            nn.Linear(config['d_model'], config['d_ff']),
            nn.GELU(),
            nn.Linear(config['d_ff'], config['d_model'])
        )
        
    def forward(self, x):
        x = x + self.dropout(self.attention(self.ln_1(x)))
        x = x + self.dropout(self.ff(self.ln_2(x)))
        return x
    
class GPTModel(nn.Module):
    def __init__(self, config):
        super(GPTModel, self).__init__()
        self.token_embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.pos_embedding = nn.Embedding(config['context_length'], config['embedding_dim'])
        self.dropout = nn.Dropout(config['dropout_rate'])

        # Adding this -> Brand new to the architecture
        self.dim_project = None
        if config['embedding_dim'] != config['d_model']:
            self.dim_project = nn.Sequential(
                nn.Linear(config['embedding_dim'], config['d_model']), 
                nn.GELU()
            )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['num_layers'])
        ])
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.fc_out = nn.Linear(config['d_model'], config['vocab_size'])

        # Tying weights
        self.fc_out.weight = self.token_embedding.weight

        # Store the config
        self.config = config

    def forward(self, x: torch.Tensor, target:torch.Tensor = None):
        batch_size, n_tokens = x.shape
        x = self.token_embedding(x) 
        x += self.pos_embedding(torch.arange(n_tokens, device=x.device)).unsqueeze(0)
        x = self.dropout(x)

        if self.dim_project:
            x = self.dim_project(x)

        for tx_block in self.transformer_blocks:
            x = tx_block(x)
        x = self.ln_f(x)
        
        if target is None:
            logits = self.fc_out(x[:, [-1], :]) # Preserve the number of dimensions
            return logits, None
        else:
            logits = self.fc_out(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100)
            return logits, loss
        
    def get_numel(self):
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
        return total_params 
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1, topk=None):
        for _ in range(max_new_tokens):
            B, T = idx.shape
            idx_curr = idx if T <= self.config['context_length'] else idx[:, -self.config['context_length']:]
            logits, loss = self(idx_curr)
            logits = logits[:, -1, :] / temperature
            if topk:
                vals, topk_idx = torch.topk(logits, k=topk, dim=-1, sorted=True)
                logits[logits < vals[:, [-1]]] = -torch.inf
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.hstack((idx, pred_idx))
        return idx