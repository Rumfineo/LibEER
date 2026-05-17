import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

from torch.autograd import Function

# 1. 定义梯度反转层 (Gradient Reversal Layer)
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def create_hankel_matrix(x, window_size: int):
    """
    x: [B, L, C]
    return: [B, L-window_size+1, window_size, C]
    """
    batch_size, seq_len, num_features = x.shape
    hankel_matrices = []
    for i in range(seq_len - window_size + 1):
        hankel_matrices.append(x[:, i:i + window_size, :])
    return torch.stack(hankel_matrices, dim=1)


class HankelMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.input_size = input_size

        self.mlp = nn.Sequential(
            nn.Linear(window_size * input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        """
        x: [B, L, C]
        return: [B, L, C]  (尾部补零对齐长度)
        """
        batch_size, seq_len, _ = x.shape
        x_hankel = create_hankel_matrix(x, self.window_size)          # [B, L', W, C]
        new_seq_len = seq_len - self.window_size + 1                  # L'
        x_reshaped = x_hankel.reshape(batch_size, new_seq_len, -1)    # [B, L', W*C]
        x_processed = self.mlp(x_reshaped)                            # [B, L', C]

        padding = torch.zeros(
            batch_size, self.window_size - 1, self.input_size, device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([x_processed, padding], dim=1)           # [B, L, C]
        return x_padded


class HankelFormer(nn.Module):
    """
    分类版 HankelFormer:
    - forward(...) 返回 (logits, repr1, repr2)
      logits: [B, num_classes]
      repr1/2: [B, d_model]  (两路 encoder 输出的 mean pooling)
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.output_attention = getattr(configs, "output_attention", False)
        self.use_norm = getattr(configs, "use_norm", True)

        # 分类数：兼容不同命名
        self.num_classes = getattr(configs, "num_classes", None)
        if self.num_classes is None:
            self.num_classes = getattr(configs, "num_class", None)
        if self.num_classes is None:
            raise ValueError("configs must provide num_classes (or num_class) for classification.")

        # Hankel MLP
        self.hankel_mlp = HankelMLP(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            window_size=configs.window_size
        )

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # 融合（保留：可用于替代/增强序列表征）
        self.fusion_layer = nn.Linear(configs.d_model * 2, configs.d_model)

        # 分类头：输出 logits
        self.classifier = nn.Linear(configs.d_model, self.num_classes, bias=True)

        self.num_domains = configs.num_domains
        self.domain_classifier = nn.Sequential(
            nn.Linear(configs.d_model, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, configs.num_domains)
        )

    def encode(self, x, x_mark):
        """
        x: [B, L, C]
        return:
          enc_out: [B, L, d_model]
          global_repr: [B, d_model]
        """
        enc_out = self.enc_embedding(x, x_mark)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        global_repr = enc_out.mean(dim=1)  # [B, d_model]
        return enc_out, global_repr

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None, alpha=1.0):
        """
        保持原工程风格的输入签名，但分类不使用 x_dec/x_mark_dec/mask。
        return: logits, repr1, repr2
        """
        # 可选归一化（分类一般未必需要；这里保持与原实现一致）
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # Hankel 分支
        x_enc_processed = self.hankel_mlp(x_enc)

        # 两路编码 + 表征
        enc_out, repr1 = self.encode(x_enc, x_mark_enc)
        enc_out_processed, repr2 = self.encode(x_enc_processed, x_mark_enc)

        # 融合后做序列池化得到最终分类表征
        fused_enc_out = self.fusion_layer(torch.cat([enc_out, enc_out_processed], dim=-1))  # [B, L, d_model]
        fused_repr = fused_enc_out.mean(dim=1)  # [B, d_model]

        logits = self.classifier(fused_repr)  # [B, num_classes]
        # 任务 2: 领域对抗 (新增)
        # 通过 GRL 翻转梯度，训练 feature extractor 模糊化受试者信息
        reverse_feature = ReverseLayerF.apply(fused_repr, alpha)
        domain_logits = self.domain_classifier(reverse_feature)
        return logits, domain_logits, repr1, repr2