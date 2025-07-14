import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import os
from MindModel.utility.logger import logger


class ObsEmbedding(nn.Module):
    def __init__(self, obs_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(obs_dim, d_model)
    def forward(self, x):  # x: [B, obs_dim]
        return self.proj(x)

class TrajActEmbedding(nn.Module):
    def __init__(self, traj_dim, action_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(traj_dim + action_dim, d_model)
    def forward(self, traj, action):
        # action should be one-hot if discrete
        x = torch.cat([traj, action], dim=-1)  # [B, seq_len, traj_dim+action_dim]
        return self.proj(x)




class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout()


        #create matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sin to even positions and cos for odd position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_len, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query, key, value, mask, dropput: nn.Dropout):
        d_k = query.shape[-1]

        #(batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=-1) #(batch, h, seq_len, seq_len)

        if dropput is not None:
            attention_score = dropput(attention_score)
        
        return (attention_score @ value), attention_score
    


    def forward(self, q, k, v, mask):
        query = self.w_q(q) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) ---> (batch, seq_len, d_model)

        #(Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)
    


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
            return self.norm(x)
        



class WorldModelOutputHead(nn.Module):
    def __init__(self, d_model, obs_dim, action_dim):
        super().__init__()
        self.obs_head = nn.Linear(d_model, obs_dim)
        self.reward_head = nn.Linear(d_model, 1)
        self.done_head = nn.Linear(d_model, 1)
        self.action_head = nn.Linear(d_model, action_dim)
    def forward(self, x):
        next_obs = self.obs_head(x)
        reward = self.reward_head(x)
        done = torch.sigmoid(self.done_head(x))
        action_logits = self.action_head(x)
        return next_obs, reward, done, action_logits



class WorldModelDistOutputHead(nn.Module):
    def __init__(self, d_model, obs_dim, action_dim):
        super().__init__()
        # For obs
        self.obs_mu = nn.Linear(d_model, obs_dim)
        self.obs_log_std = nn.Linear(d_model, obs_dim)
        # For reward
        self.reward_mu = nn.Linear(d_model, 1)
        self.reward_log_std = nn.Linear(d_model, 1)
        # For done (binary)
        self.done_head = nn.Linear(d_model, 1)
        # For action (categorical)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, x):
        # [B, seq_len, d_model] -> all outputs [B, seq_len, ...]
        obs_mu = self.obs_mu(x)
        obs_log_std = self.obs_log_std(x).clamp(-10, 2)   # clamp for stability
        reward_mu = self.reward_mu(x)
        reward_log_std = self.reward_log_std(x).clamp(-10, 2)
        done = torch.sigmoid(self.done_head(x))
        action_logits = self.action_head(x)
        return obs_mu, obs_log_std, reward_mu, reward_log_std, done, action_logits




class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

    

class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    



class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
    
    
class TransformerWorldModel(nn.Module):
    def __init__(
        self, 
        obs_dim, action_dim, d_model=256, N=4, h=8, dropout=0.1, d_ff=512, max_len=32
    ):
        super().__init__()
        # Embeddings
        self.obs_embed = ObsEmbedding(obs_dim, d_model)
        self.trajact_embed = TrajActEmbedding(obs_dim + 2, action_dim, d_model)
        self.src_pos = PositionalEncoding(d_model, max_len, dropout)
        self.tgt_pos = PositionalEncoding(d_model, max_len, dropout)
        # Encoder
        encoder_blocks = nn.ModuleList([
            EncoderBlock(
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout
            ) for _ in range(N)
        ])
        self.encoder = Encoder(encoder_blocks)
        # Decoder
        decoder_blocks = nn.ModuleList([
            DecoderBlock(
                MultiHeadAttentionBlock(d_model, h, dropout),
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout
            ) for _ in range(N)
        ])
        self.decoder = Decoder(decoder_blocks)
        # Output heads
        self.output_head = WorldModelOutputHead(d_model, obs_dim, action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, obs, traj_seq, action_seq, src_mask=None, tgt_mask=None):
        """
        obs:        [B, obs_dim]
        traj_seq:   [B, seq_len, obs_dim+2]   (reward, done included)
        action_seq: [B, seq_len, action_dim]  (one-hot or real)
        """
        src = self.src_pos(self.obs_embed(obs).unsqueeze(1))  # [B, 1, d_model]
        tgt = self.tgt_pos(self.trajact_embed(traj_seq, action_seq))  # [B, seq_len, d_model]
        memory = self.encoder(src, src_mask)  # [B, 1, d_model]
        dec_out = self.decoder(tgt, memory, src_mask, tgt_mask)  # [B, seq_len, d_model]
        next_obs, reward, done, action_logits = self.output_head(dec_out)
        # All outputs are [B, seq_len, ...]
        return next_obs, reward, done, action_logits
    

    def save_model(self, folder: str = "MindModel_version", filename: str = "transformer_model.pt"):
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        torch.save({
            'model_state_dict': self.state_dict(),
        }, save_path)
        logger.info(f"[✓] TransformerWorldModel saved at: {save_path}")

    def load_model(self, folder: str = "checkpoints", filename: str = "transformer_model.pt"):
        load_path = os.path.join(folder, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"[✗] Model checkpoint not found at: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"[✓] TransformerWorldModel loaded from: {load_path}")



class ProbabilisticTransformerWorldModel(nn.Module):
    def __init__(
        self, 
        obs_dim, action_dim, d_model=256, N=4, h=8, dropout=0.1, d_ff=512, max_len=32
    ):
        super().__init__()
        # Embeddings
        self.obs_embed = ObsEmbedding(obs_dim, d_model)
        self.trajact_embed = TrajActEmbedding(obs_dim + 2, action_dim, d_model)
        self.src_pos = PositionalEncoding(d_model, max_len, dropout)
        self.tgt_pos = PositionalEncoding(d_model, max_len, dropout)
        # Encoder
        encoder_blocks = nn.ModuleList([
            EncoderBlock(
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout
            ) for _ in range(N)
        ])
        self.encoder = Encoder(encoder_blocks)
        # Decoder
        decoder_blocks = nn.ModuleList([
            DecoderBlock(
                MultiHeadAttentionBlock(d_model, h, dropout),
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout
            ) for _ in range(N)
        ])
        self.decoder = Decoder(decoder_blocks)
        # Output heads
        self.output_head = WorldModelDistOutputHead(d_model, obs_dim, action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, obs, traj_seq, action_seq, src_mask=None, tgt_mask=None):
        """
        Standard forward: predict parameters of next-step distributions given context.
        """
        src = self.src_pos(self.obs_embed(obs).unsqueeze(1))  # [B, 1, d_model]
        tgt = self.tgt_pos(self.trajact_embed(traj_seq, action_seq))  # [B, seq_len, d_model]
        memory = self.encoder(src, src_mask)  # [B, 1, d_model]
        dec_out = self.decoder(tgt, memory, src_mask, tgt_mask)  # [B, seq_len, d_model]
        # Distributional outputs:
        obs_mu, obs_log_std, reward_mu, reward_log_std, done, action_logits = self.output_head(dec_out)
        return obs_mu, obs_log_std, reward_mu, reward_log_std, done, action_logits
    
    def sample(self, obs_mu, obs_log_std):
        std = torch.exp(obs_log_std)
        return obs_mu + std * torch.randn_like(std)


    

    def save_model(self, folder: str = "MindModel_version", filename: str = "probabilistic_transformer_model.pt"):
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        torch.save({
            'model_state_dict': self.state_dict(),
        }, save_path)
        logger.info(f"[✓] TransformerWorldModel saved at: {save_path}")

    def load_model(self, folder: str = "checkpoints", filename: str = "probabilistic_transformer_model.pt"):
        load_path = os.path.join(folder, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"[✗] Model checkpoint not found at: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"[✓] TransformerWorldModel loaded from: {load_path}")