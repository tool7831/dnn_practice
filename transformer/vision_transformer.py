import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # B, H, N, Hc
        
        attn = (q @ k.transpose(-2, -1)) / self.scale # B, H, N, N
        attn = F.softmax(attn, dim=-1)
        
        x = attn @ v # B, H, N, Hc
        x = x.transpose(2,1) # B, N, H, Hc
        x = x.reshape(B,N,C) # B, N, C
        x = self.proj(x)
        
        return x
    
    
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_rate=0.1):
        super(MultiLayerPerceptron, self).__init__()
        
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, input_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self, x):
        x = self.dropout(self.gelu(self.w1(x)))
        x = self.dropout(self.w2(x))
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, drop_rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.msa = MultiHeadSelfAttention(input_dim, num_heads)
        self.norm2 = nn.LayerNorm(input_dim)
        self.mlp = MultiLayerPerceptron(input_dim, hidden_dim, drop_rate)
        
    def forward(self, x):
        x = self.msa(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x
        
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x) # B, C, Ph, Pw
        x = x.flatten(2) # B, C , Number of patches
        x = x.transpose(1,2) # B, N, C
        return x
        
        
class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, in_channels=3, embed_dim=768, num_heads=12, blocks=12, mlp_dim=3072, dropout_rate=0.1):
        super(ViT, self).__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros((1, num_patches + 1, embed_dim)), requires_grad=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.transformer = nn.Sequential(
            *[TransformerBlock(input_dim=embed_dim, num_heads=num_heads, hidden_dim=mlp_dim, drop_rate=dropout_rate) for i in range(blocks)]
        )
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        '''
            x: (B, in_channels, H, W)
        '''
        B = x.shape[0]
        x = self.patch_embed(x) # B, number of patches, embde_dim
        cls_token = self.cls_token.expand(B, -1, -1) # B, 1, embed_dim
        x = torch.cat((cls_token, x),dim=1) # B, num patches + 1, embed_dim
        
        x = x + self.pos_embed
        x = self.dropout(x)
        
        x = self.transformer(x)

        x = x[:, 0] # class token
        x = self.head(x)
        return x
    
def vit_b(image_size=224, patch_size=16, in_channels=3, num_classes=1000):
    return ViT(image_size=image_size, patch_size=patch_size, 
               num_classes=num_classes, in_channels=in_channels, 
               embed_dim=768, blocks=12, num_heads=12, mlp_dim=3072)
    
def vit_l(image_size=224, patch_size=16, in_channels=3, num_classes=1000):
    return ViT(image_size=image_size, patch_size=patch_size, 
               num_classes=num_classes, in_channels=in_channels, 
               embed_dim=1024, blocks=24, num_heads=16, mlp_dim=4096)
    
def vit_h(image_size=224, patch_size=16, in_channels=3, num_classes=1000):
    return ViT(image_size=image_size, patch_size=patch_size, 
               num_classes=num_classes, in_channels=in_channels, 
               embed_dim=1280, blocks=32, num_heads=16, mlp_dim=5120)