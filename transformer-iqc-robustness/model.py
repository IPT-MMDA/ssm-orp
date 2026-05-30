import torch
import torch.nn as nn
import torch.nn.functional as F


class VerifiableAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        self.W_q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, head_dim, bias=False)


    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)

        return out, scores, attn_weights


if __name__ == "__main__":
    torch.manual_seed(1)

    SEQ_LEN = 3
    EMBED_DIM = 4
    HEAD_DIM = 4

    model = VerifiableAttention(embed_dim=EMBED_DIM, head_dim=HEAD_DIM)
    x_clean = torch.randn(SEQ_LEN, EMBED_DIM)
    out, pre_softmax, post_softmax = model(x_clean)

    print(x_clean)
    print(pre_softmax)
    print(post_softmax)
