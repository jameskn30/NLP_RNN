from torch import nn
import torch


class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.5, qkv_bias=False) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d out must be divisible by num_heads"

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.d_in = d_in
        self.d_out = d_out
        self.head_dim = d_out // num_heads
        self.num_heads = num_heads

        self.out_proj = nn.Linear(d_out, d_out)

        nn.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(
            torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        '''
        params:
            x: input sequence (batch, num_tokens, d_in)

        returns:
            y: context vector (batch, num_tokens, d_out)
        '''

        batch, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # split heads
        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim)

        # transpose
        # shape = (batch, num_heads, num_tokens , head_dim)
        keys = keys.transpose(1, 2)
        # shape = (batch, num_heads, num_tokens , head_dim)
        queries = queries.transpose(1, 2)
        # shape = (batch, num_heads, num_tokens , head_dim)
        values = values.transpose(1, 2)

        # do batch matmul
        # NOTE: can we improve this with torch.bmm ?
        # shape = (batch, num_heads , num_tokens, num_tokens)
        attn_scores: torch.Tensor = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # shape = (batch, num_heads, num_tokens, num_tokens)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # shape = (batch, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values)
        # shape = (batch, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)

        context_vec = context_vec.contiguous().view(batch, num_tokens, self.d_out)

        context_vec = self.out_proj(context_vec)

        return context_vec


class FeedForward(nn.Module):

    def __init__(self, d_in) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_in, 4 * d_in),
            nn.GELU(),
            nn.Linear(4 * d_in, d_in),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        vocab_size = cfg['vocab_size']
        context_length = cfg['context_length']
        embed_dim = cfg['embed_dim']
        n_heads = cfg['n_heads']
        n_layers = cfg['n_layers']
        drop_rate = cfg['drop_rate']
        qkv_bias = cfg['qkv_bias']

        self.ff = FeedForward(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim, embed_dim, context_length, n_heads, drop_rate, qkv_bias=qkv_bias)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)

        x = x + shortcut

        shortcut = x

        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)

        x = x + shortcut
        return x


class GPTModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.vocab_size = cfg['vocab_size']
        self.context_length = cfg['context_length']
        self.embed_dim = cfg['embed_dim']
        self.n_heads = cfg['n_heads']
        self.n_layers = cfg['n_layers']
        self.drop_rate = cfg['drop_rate']
        self.qkv_bias = cfg['qkv_bias']
        self.eos_id = cfg['eos_id']

        self.token_emb = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_emb = nn.Embedding(self.context_length, self.embed_dim)
        self.drop_emb = nn.Dropout(self.drop_rate)

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(self.n_layers)])
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.out_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch, num_tokens = x.shape

        token_embeddings = self.token_emb(x)

        pos_embeddings = self.pos_emb(
            torch.arange(num_tokens, device=x.device))

        x = token_embeddings + pos_embeddings
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def generate_text_simple(self, ids, max_new_tokens, context_size):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = ids[:, -context_size:]

            with torch.no_grad():
                logits = self(idx_cond)

            logits = logits[:, -1, :]
            # The softmax function is monotonic, meaning it preserves the order of its inputs when transformed into outputs
            # not neccessary. Explained in chapt 4 page 144
            probas = torch.softmax(logits, dim=-1)

            idx_next = torch.argmax(probas, dim=-1, keepdim=True)

            ids = torch.cat((ids, idx_next), dim=-1)

        self.train()

        return ids

    def generate(self, ids, max_new_tokens, context_size, temperature = 0.0, top_k = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = ids[:, -context_size:]

            with torch.no_grad():
                logits = self(idx_cond)
            
            logits = logits[:, -1, :]

            if top_k:
                #use top_k sampling
                top_logits, top_pos = torch.topk(logits, top_k)

                logits = torch.where(
                    logits < top_logits[:,-1],
                    input = torch.tensor(-float('inf')).to(logits.device), #if condition is true
                    other = logits #if condition is false
                )
            
            if temperature > 0.0:
                #use temp scaling
                logits = logits / temperature
                probs = torch.softmax(logits, dim = -1)
                # sample from probability distribution
                idx_next = torch.multinomial(probs, num_samples=1) 
            else:
                idx_next = torch.argmax(logits, dim = -1, keepdim = True) 
            
            # if idx_next == EOS_ID: # stop generaton if EOS is predicted
            #     break
                
            ids = torch.cat((ids, idx_next), dim = 1)
        
        return ids
