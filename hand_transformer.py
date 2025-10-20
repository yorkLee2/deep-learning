import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现
    输入: q, k, v 形状为 (batch_size, seq_len, d_model)
    输出: 注意力输出和注意力权重
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "模型维度必须能被注意力头数整除"
        
        # 模型参数
        self.d_model = d_model                    # 模型总维度
        self.num_heads = num_heads                # 注意力头数量
        self.head_dim = d_model // num_heads      # 每个头的维度

        # 定义线性变换层
        self.w_q = nn.Linear(d_model, d_model, bias=False)    # Q 变换
        self.w_k = nn.Linear(d_model, d_model, bias=False)    # K 变换  
        self.w_v = nn.Linear(d_model, d_model, bias=False)    # V 变换
        
        # 输出投影层
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 正则化
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _split_heads(self, x):
        """
        将输入张量分割成多个注意力头
        输入: (batch_size, seq_len, d_model)
        输出: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 重塑张量以分离注意力头
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 调整维度顺序以便批量矩阵乘法
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x):
        """
        将多个注意力头的输出合并回原始维度
        输入: (batch_size, num_heads, seq_len, head_dim)
        输出: (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # 调整维度顺序以便合并
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        
        # 合并所有注意力头
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, q, k, v, mask=None):
        """
        前向传播
        q, k, v: 形状为 (batch_size, seq_len, d_model)
        mask: 注意力掩码，形状可广播到 (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_q, _ = q.shape
        _, seq_k, _ = k.shape

        # 步骤1: 线性投影得到 Q, K, V
        Q = self.w_q(q)  # (batch_size, seq_q, d_model)
        K = self.w_k(k)  # (batch_size, seq_k, d_model)  
        V = self.w_v(v)  # (batch_size, seq_k, d_model)

        # 步骤2: 分割成多个注意力头
        Q_heads = self._split_heads(Q)  # (batch_size, num_heads, seq_q, head_dim)
        K_heads = self._split_heads(K)  # (batch_size, num_heads, seq_k, head_dim)
        V_heads = self._split_heads(V)  # (batch_size, num_heads, seq_k, head_dim)

        # 步骤3: 计算缩放点积注意力分数
        # Q * K^T / sqrt(head_dim)
        scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) 
        scores = scores / math.sqrt(self.head_dim)

        # 步骤4: 应用注意力掩码（如果提供）
        if mask is not None:
            # 确保掩码是布尔类型，True表示允许注意力，False表示屏蔽
            if mask.dtype != torch.bool:
                mask_bool = mask.to(torch.bool)
            else:
                mask_bool = mask
            
            # 将屏蔽位置的值设为负无穷，这样softmax后权重为0
            scores = scores.masked_fill(~mask_bool, float("-1e9"))

        # 步骤5: 计算注意力权重
        attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_q, seq_k)
        attn_weights = self.dropout(attn_weights)  # 应用dropout

        # 步骤6: 应用注意力权重到值向量
        # attn_weights * V
        context = torch.matmul(attn_weights, V_heads)  # (batch_size, num_heads, seq_q, head_dim)

        # 步骤7: 合并所有注意力头
        combined = self._combine_heads(context)  # (batch_size, seq_q, d_model)
        
        # 步骤8: 最终线性投影
        output = self.out_proj(combined)  # (batch_size, seq_q, d_model)

        return output, attn_weights


# --- 掩码生成函数 ---
def make_causal_mask(seq_len, device=None):
    """
    创建因果掩码（用于自回归任务）
    返回下三角矩阵，True表示允许注意力，False表示屏蔽
    形状: (seq_len, seq_len)
    """
    # 创建上三角矩阵（对角线以上为True）
    upper_triangle = torch.triu(torch.ones((seq_len, seq_len), 
                                         dtype=torch.bool, device=device), 
                              diagonal=1)
    # 取反得到下三角矩阵（含对角线）
    return ~upper_triangle  # 下三角（含对角线）为True


def make_padding_mask(pad_positions, device=None):
    """
    创建填充掩码
    pad_positions: (batch_size, seq_len) 布尔张量，True表示填充位置
    返回: (batch_size, 1, 1, seq_len) 布尔张量，True表示允许注意力
    """
    if pad_positions.dtype != torch.bool:
        pad_bool = pad_positions.to(torch.bool)
    else:
        pad_bool = pad_positions
    
    # 非填充位置为True（允许注意力）
    allowed = ~pad_bool  # (batch_size, seq_len)
    
    # 扩展维度以匹配注意力分数的形状
    return allowed.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)


# --- 演示和测试 ---
if __name__ == "__main__":
    # 设置随机种子以便重现结果
    torch.manual_seed(42)
    
    # 模型参数
    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4

    # 创建多头注意力模块
    mha = MultiHeadAttention(
        d_model=d_model, 
        num_heads=num_heads, 
        dropout=0.1
    )

    # 创建随机输入数据
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    print("输入形状:")
    print(f"q: {q.shape}")
    print(f"k: {k.shape}") 
    print(f"v: {v.shape}")

    # 测试1: 无掩码情况
    print("\n=== 测试1: 无注意力掩码 ===")
    output_no_mask, attn_no_mask = mha(q, k, v)
    print(f"输出形状: {output_no_mask.shape}")
    print(f"注意力权重形状: {attn_no_mask.shape}")

#     # 测试2: 使用填充掩码
#   #  print("\n=== 测试2: 使用填充掩码 ===")
#     # 创建填充指示器：第一个样本最后两个位置是填充
#  #   pad = torch.tensor([
#  #       [0, 0, 0, 1, 1],  # 第一个样本：位置3,4是填充
#  #       [0, 0, 0, 0, 0]   # 第二个样本：无填充
#   #  ])
    
#   #  pad_mask = make_padding_mask(pad)
#  #   output_pad, attn_pad = mha(q, k, v, mask=pad_mask)
#  #   print(f"输出形状: {output_pad.shape}")
    
#     # 显示第一个样本第一个头的注意力权重（应该看到最后两列为0）
#  #   print("第一个样本第一个头的注意力权重（最后两列应该接近0）:")
#   #  print(attn_pad[0, 0].detach().numpy().round(3))

#     # 测试3: 使用因果掩码
#  #   print("\n=== 测试3: 使用因果掩码 ===")
#     causal_mask = make_causal_mask(seq_len)
#     # 扩展维度以匹配注意力分数的形状
#     causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
#     output_causal, attn_causal = mha(q, k, v, mask=causal_mask)
#     print(f"输出形状: {output_causal.shape}")
    
#     # 显示因果掩码的注意力权重（应该是下三角模式）
#     print("因果掩码下的注意力权重（上三角应该为0）:")
#     print(attn_causal[0, 0].detach().numpy().round(3))

#     print("\n所有测试完成！多头注意力机制工作正常。")