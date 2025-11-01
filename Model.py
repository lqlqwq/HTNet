from functools import partial
import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

# 创建元组(一个不可修改的序列)
def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

# 归一化 
class LayerNorm(nn.Module):
    # 设置g和b偏移参数,g用于调整尺度,b用于调整位置,eps用于防止除0错误  nn.Parameter代表这是需要训练的参数
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    # B(哪张图),C(那张图的哪个特征通道),H(具体坐标),W(具体坐标) 相当于4个索引,精确到一个数值
    # Dim=1代表跨通道归一化,方式为正态分布,方差使用有偏估计(/N)
    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# 在Attention和FeedForward之前进行自定义归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    # 类似于重写了方法？在调用fn之前先调用归一化
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 前馈神经网络 MLP=Multi Layer Perceptron
# 升维-激活-Dropout-降维-Dropout
# Dropout可选,随机丢弃部分特征,防止过拟合
# Gelu 激活函数,使线性变为非线性来达到拟合
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# 注意力层 从总dim转化为多头小dim,从多个方面进行学习
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        # 特征数/头数=每个头特征数
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        # 等价于1 / math.sqrt(dim_head)
        self.scale = dim_head ** -0.5

        # Softmax对dim=-1(最后一维)进行归一化(非线性,放大大值,缩小小值,且使总和为1)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # *拆包写法,自动匹配变量数
        b, c, h, w, heads = *x.shape, self.heads

        # 沿着dim=1(特征层)把qkv拆成3个chunk
        qkv = self.to_qkv(x).chunk(3, dim = 1)

        # to_qkv把特征维度从dim转为inner_dim, inner_dim = heads*dim_head = (h d)
        # 把完整的通道拆成多头小通道,把长宽的空间拉直成一个序列, 按照每个像素点计算相似度
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), qkv)

        # 计算q和k的点积,得到相似度矩阵 n=x*y, i=j=n(一个是查询矩阵一个是键矩阵), 计算查询矩阵的键矩阵的相似度
        # self.scale进行缩放,缩小数量级差距
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Softmax归一化+Dropout
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 恢复形状
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# 升特征维度+缩小空间尺寸
def Aggregate(dim, dim_out):
    return nn.Sequential(
        # 卷积,这里扩大特征维度
        nn.Conv2d(dim, dim_out, 3, padding = 1),
        LayerNorm(dim_out),
        # Stride=2,池化层,空间缩小一半
        nn.MaxPool2d(3, stride = 2, padding = 1)
    )

class Transformer(nn.Module):
    # seq_len,序列长度,表示被分割为多少块; depth, 深度, 表示有多少Transformer层; mlp_mult, 前馈神经网络中特征维度的倍数, 表示特征扩大几倍
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        # 初始化一个可学习的位置参数,用来突出某些位置的重要性
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        # 有多少层就加多少注意力和前馈
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))

    # 自动调用  
    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h = h, w = w)
        x = x + pos_emb

        # 在Modulelist里以数组形式储存,这里直接对应数组内第一个和第二个元素
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# HTNet的作用: 接收参数，遍历3个层级，调用对应的函数创建任务
class HTNet(nn.Module):
    # image_size=28, patch_size=7, dim=256, heads=3, num_hierarchies=3, block_repeats=(2, 2, 10) num_classes=3
    # channels指的是图像的输入通道数,RGB图像为3通道
    # *后的参数必须显式传参,不能顺序传参
    def __init__(
        self,
        *,
        image_size, # 图片尺寸27
        patch_size, # 小块尺寸7
        num_classes, # 3分类
        dim, # 特征个数=256
        heads, # 特征头数 实际3默认8？
        num_hierarchies, # 特征层数=3
        block_repeats, # 特征层重复次数 (2,2,10)
        mlp_mult = 4, # 前馈卷积时的特征放大倍数
        channels = 3, # 图片通道数,RGB 3通道
        dim_head = 64, # 每个头的特征个数
        dropout = 0.
    ):
        # nn.Module 初始化
        super().__init__()

        # 验证图片尺寸关系
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'

        # 一些参数计算
        patch_dim = channels * patch_size ** 2
        fmap_size = image_size // patch_size # 4
        blocks = 2 ** (num_hierarchies - 1) # 4
        seq_len = (fmap_size // blocks) ** 2   # sequence length is held constant across heirarchy

        # 3层,按照2,1,0放入数组
        hierarchies = list(reversed(range(num_hierarchies)))
        # 再次反转, 让浅层的倍率低,深层的倍率高
        mults = [2 ** i for i in reversed(hierarchies)] # [1, 2, 4]

        # Map把一个函数依次应用到一个可迭代对象上,这里对mults数组的每个元素都引用了匿名函数lambda
        layer_heads = list(map(lambda t: t * heads, mults)) # [3, 6, 12]
        layer_dims = list(map(lambda t: t * dim, mults)) # [256, 512, 1024]
        last_dim = layer_dims[-1] # 1024

        # 创建输入维度和输出维度，最后一层是输出层不用升维所以保持不变
        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])

        # 按patch_size切割特征图, 并非实际图片, 然后卷积成目标特征数
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])

        # 使用Zip同步遍历多数组,自动配对,按最短长度阶段,甚至还能自动解包,效率高, 但是Zip的输出是不可变的
        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat
            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))

        # 分类
        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level # [4,2,1]
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
            x = aggregate(x)
        return self.mlp_head(x)

# # This function is to confuse three models
# class Fusionmodel(nn.Module):
#   def __init__(self):
#     #  extend from original
#     super(Fusionmodel,self).__init__()
#     self.fc1 = nn.Linear(15, 3)
#     self.bn1 = nn.BatchNorm1d(3)
#     self.d1 = nn.Dropout(p=0.5)
#     self.fc_2 = nn.Linear(6, 3)
#     self.relu = nn.ReLU()
#     # forward layers is to use these layers above
#   def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
#     fuse_four_features = torch.cat((l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature), 0)
#     fuse_out = self.fc1(fuse_four_features)
#     fuse_out = self.relu(fuse_out)
#     fuse_out = self.d1(fuse_out) # drop out
#     fuse_whole_four_parts = torch.cat(
#         (whole_feature,fuse_out), 0)
#     fuse_whole_four_parts = self.relu(fuse_whole_four_parts)
#     fuse_whole_four_parts = self.d1(fuse_whole_four_parts)
#     out = self.fc_2(fuse_whole_four_parts)
#     return out
