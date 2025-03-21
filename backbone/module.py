import torch
import torch.nn as nn
import torch.nn.functional as F


# class DomainTransformationModule(nn.Module):
#     def __init__(self, feature_dim, num_heads=4):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.num_heads = num_heads
#         self.embedding = nn.Linear(feature_dim, feature_dim)
#         self.attention = nn.MultiheadAttention(feature_dim, num_heads)
#         self.residual_fc = nn.Linear(feature_dim, feature_dim)

#     def reset(self):
#         self.embedding = nn.Linear(self.feature_dim, self.feature_dim)
#         self.attention = nn.MultiheadAttention(self.feature_dim, self.num_heads)
#         self.residual_fc = nn.Linear(self.feature_dim, self.feature_dim)

#     def forward(self, x, prototypes, noise_scale=None):
#         # x: [batch_size, feature_dim] → [seq_len=1, batch_size, feature_dim]
#         e_x = self.embedding(x).unsqueeze(0)  # [1, batch_size, feature_dim]
        
#         # prototypes: [num_prototypes, feature_dim] → [seq_len=num_prototypes, batch_size, feature_dim]
#         e_p = self.embedding(prototypes)  # [num_prototypes, feature_dim]
#         e_p = e_p.unsqueeze(1).expand(-1, x.size(0), -1)  # [num_prototypes, batch_size, feature_dim]

#         attn_output, attn_weights = self.attention(
#             query=e_x,       # [1, batch_size, feature_dim]
#             key=e_p,         # [num_prototypes, batch_size, feature_dim]
#             value=e_p        # [num_prototypes, batch_size, feature_dim]
#         )
#         attn_output = attn_output.squeeze(0)  # [batch_size, feature_dim]

#         if noise_scale is not None:
#             noise = noise_scale * torch.randn_like(attn_output)
#             attn_output = attn_output + noise
#         augmented_feature = F.relu(x + self.residual_fc(attn_output))
#         return augmented_feature
    

class DomainTransformationModule(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.reset()

    def reset(self):
        self.pre_embed_norm = nn.LayerNorm(self.feature_dim)
        # FIXME embedding layers may not be necessary
        self.semantic_embed = nn.Linear(self.feature_dim, self.feature_dim)
        self.domain_embed = nn.Linear(self.feature_dim, self.feature_dim)
        self.attention = nn.MultiheadAttention(self.feature_dim, self.num_heads)
        self.post_attn_norm = nn.LayerNorm(self.feature_dim)
        self.readout = nn.Linear(self.feature_dim, self.feature_dim)
        self.act = nn.GELU()

    def forward(self, x:torch.Tensor, k:torch.Tensor, v:torch.Tensor=None):
        '''
            x: input feature, final layer, size [batch_size, feature_dim]
            k: inner layer prototype: [batch_size, num_layers, feature_dim]
        '''
        # x: [batch_size, feature_dim] → [seq_len=1, batch_size, feature_dim]
        e_x = self.pre_embed_norm(x)
        e_x = self.semantic_embed(e_x).unsqueeze(0)  # [1, batch_size, feature_dim]

        # keys: [batch_size, num_keys, feature_dim] → [seq_len=num_keys, batch_size, feature_dim]
        e_k = self.pre_embed_norm(k)
        e_k = self.domain_embed(e_k).permute(1, 0, 2)  # [num_keys, batch_size, feature_dim]

        # values: [batch_size, num_keys, feature_dim] → [seq_len=num_keys, batch_size, feature_dim]
        if v is None:
            e_v = e_k
        else:
            e_v = self.pre_embed_norm(v)
            e_v = self.domain_embed(e_v).permute(1, 0, 2)

        attn_output, attn_weights = self.attention(
            query= e_x,         # [1, batch_size, feature_dim]
            key  = e_k,         # [num_keys, batch_size, feature_dim]
            value= e_v          # [num_keys, batch_size, feature_dim]
        )
        attn_output = attn_output.squeeze(0)  # [batch_size, feature_dim]
        attn_output = self.post_attn_norm(attn_output)

        augmented_feature = x + self.readout(attn_output)
        augmented_feature = self.act(augmented_feature)
        return augmented_feature
    

class DomainTransformationModuleV2(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.embedding = nn.Linear(feature_dim, feature_dim)
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, dropout=0.1)
        self.residual_fc_attn = nn.Linear(feature_dim, feature_dim)
        self.residual_fc_x = nn.Linear(feature_dim, feature_dim)
        self.act = nn.GELU()

    def reset(self):
        self.embedding = nn.Linear(self.feature_dim, self.feature_dim)
        self.attention = nn.MultiheadAttention(self.feature_dim, self.num_heads, dropout=0.1)
        self.residual_fc_attn = nn.Linear(self.feature_dim, self.feature_dim)
        self.residual_fc_x = nn.Linear(self.feature_dim, self.feature_dim)
        self.act = nn.GELU()

    def forward(self, x, prototypes):
        # x: [batch_size, feature_dim] → [seq_len=1, batch_size, feature_dim]
        e_x = self.embedding(x).unsqueeze(0)  # [1, batch_size, feature_dim]
        
        # prototypes: [num_prototypes, feature_dim] → [seq_len=num_prototypes, batch_size, feature_dim]
        e_p = self.embedding(prototypes)  # [num_prototypes, feature_dim]
        e_p = e_p.unsqueeze(1).expand(-1, x.size(0), -1)  # [num_prototypes, batch_size, feature_dim]

        attn_output, attn_weights = self.attention(
            query=e_x,       # [1, batch_size, feature_dim]
            key=e_p,         # [num_prototypes, batch_size, feature_dim]
            value=e_p        # [num_prototypes, batch_size, feature_dim]
        )
        attn_output = attn_output.squeeze(0)  # [batch_size, feature_dim]

        augmented_feature = self.act(self.residual_fc_x(x) + self.residual_fc_attn(attn_output))
        return augmented_feature
    

class FeatMatch(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.embedding = nn.Linear(feature_dim, feature_dim)
        self.aggregate = nn.Linear(self.head_dim * 2, self.head_dim)
        self.readout = nn.Linear(feature_dim, feature_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def reset(self):
        self.embedding = nn.Linear(self.feature_dim, self.feature_dim)
        self.aggregate = nn.Linear(self.head_dim * 2, self.head_dim)
        self.readout = nn.Linear(self.feature_dim, self.feature_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, prototypes):
        # x: [batch_size, feature_dim]
        # prototypes: [num_prototypes, feature_dim]
        e_x = self.embedding(x)  # [batch_size, feature_dim]
        e_p = self.embedding(prototypes)  # [num_prototypes, feature_dim]

        e_x = e_x.view(x.size(0), self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, batch_size, head_dim]
        e_p = e_p.view(prototypes.size(0), self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, num_prototypes, head_dim]

        attn_weights = torch.matmul(e_x, e_p.transpose(1, 2)) / self.head_dim ** 0.5  # [num_heads, batch_size, num_prototypes]
        attn_weights = F.softmax(attn_weights, dim=-1)  # [num_heads, batch_size, num_prototypes]
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, e_p)  # [num_heads, batch_size, head_dim]
        ex_attn = torch.cat([e_x, attn_output], dim=-1) # [num_heads, batch_size, 2*head_dim]
        agg_output = self.aggregate(ex_attn) # [num_heads, batch_size, head_dim]
        agg_output = agg_output.transpose(0, 1).contiguous().view(x.size(0), -1)  # [batch_size, feature_dim]

        output = self.readout(agg_output)  # [batch_size, feature_dim]
        output = self.act(x + output)

        return output
        