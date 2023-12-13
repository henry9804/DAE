# G 改 E, 实际上需要用G Block改出E block, 完成逆序对称，在同样位置还原style潜码
# 比第0版多了残差, 每一层的两个(conv/line)输出的w1和w2合并为1个w
# 比第1版加了要学习的bias_1和bias_2，网络顺序和第1版有所不同(更对称)
# 比第2版，即可以使用到styleganv1,styleganv2, 不再使用带Equalize learning rate的Conv (这条已经废除). 以及Block第二层的blur操作
# 改变了上采样，不在conv中完成
# 改变了In,带参数的学习
# 改变了了residual,和残差网络一致，另外旁路多了conv1处理通道和In学习参数
# 经测试，不带Eq(Equalize Learning Rate)的参数层学习效果不好

#这一版兼容PGGAN和BIGGAN: 主要改变最后一层，增加FC
#PGGAN: 加一个fc, 和原D类似
#BIGGAN,加两个fc，各128channel，其中一个是标签，完成128->1000的映射
#BIGGAN 改进思路1: IN替换CBN (本例实现)
#BIGGAN 改进思路2: G加w，和E的w对称 (未实现)
import math
import torch
import torch.nn as nn 
from torch.nn import init
from torch.nn.parameter import Parameter
import sys
sys.path.append('../')
from torch.nn import functional as F
import model.utils.lreq as ln

# G 改 E, 实际上需要用G Block改出E block, 完成逆序对称，在同样位置还原style潜码
# 比第0版多了残差, 每一层的两个(conv/line)输出的w1和w2合并为1个w
# 比第1版加了要学习的bias_1和bias_2，网络顺序和第1版有所不同(更对称)

def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)

class BigGANBatchNorm(nn.Module):
    """ This is a batch norm module that can handle conditional input and can be provided with pre-computed
        activation means and variances for various truncation parameters.

        We cannot just rely on torch.batch_norm since it cannot handle
        batched weights (pytorch 1.0.1). We computate batch_norm our-self without updating running means and variances.
        If you want to train this model you should add running means and variance computation logic.
    """
    def __init__(self, num_features, condition_vector_dim=None, n_stats=51, eps=1e-4, conditional=True):
        super(BigGANBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.conditional = conditional

        # We use pre-computed statistics for n_stats values of truncation between 0 and 1
        self.register_buffer('running_means', torch.zeros(n_stats, num_features))
        self.register_buffer('running_vars', torch.ones(n_stats, num_features))
        self.step_size = 1.0 / (n_stats - 1)

        if conditional:
            assert condition_vector_dim is not None
            self.scale = snlinear(in_features=condition_vector_dim, out_features=num_features, bias=False, eps=eps)
            self.offset = snlinear(in_features=condition_vector_dim, out_features=num_features, bias=False, eps=eps)
        else:
            self.weight = torch.nn.Parameter(torch.Tensor(num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_features))

    def forward(self, x, truncation, condition_vector=None):
        # Retreive pre-computed statistics associated to this truncation
        coef, start_idx = math.modf(truncation / self.step_size)
        start_idx = int(start_idx)
        if coef != 0.0:  # Interpolate
            running_mean = self.running_means[start_idx] * coef + self.running_means[start_idx + 1] * (1 - coef)
            running_var = self.running_vars[start_idx] * coef + self.running_vars[start_idx + 1] * (1 - coef)
        else:
            running_mean = self.running_means[start_idx]
            running_var = self.running_vars[start_idx]

        if self.conditional:
            running_mean = running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            running_var = running_var.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            weight = 1 + self.scale(condition_vector).unsqueeze(-1).unsqueeze(-1)
            bias = self.offset(condition_vector).unsqueeze(-1).unsqueeze(-1)

            out = (x - running_mean) / torch.sqrt(running_var + self.eps) * weight + bias
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias,
                               training=False, momentum=0.0, eps=self.eps)
        return out

class FromRGB(nn.Module):
    def __init__(self, channels, outputs):
        super(FromRGB, self).__init__()
        self.from_rgb = torch.nn.Conv2d(channels, outputs, 1, 1, 0)
    def forward(self, x):
        x = self.from_rgb(x)
        x = F.leaky_relu(x, 0.2)
        return x

class BEBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, has_second_conv=True, fused_scale=True): #分辨率大于128用fused_scale,即conv完成上采样
        super().__init__()
        self.has_second_conv = has_second_conv
        self.noise_weight_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.noise_weight_1.data.zero_()
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        #self.instance_norm_1 = nn.InstanceNorm2d(inputs, affine=True, eps=1e-8)
        self.batch_norm_1 = BigGANBatchNorm(inputs, condition_vector_dim=256, n_stats=51, eps=1e-12, conditional=True) 
        #self.inver_mod1 = ln.Linear(2 * inputs, latent_size) # [n, 2c] -> [n,512]
        self.conv_1 = ln.Conv2d(inputs, inputs, 3, 1, 1, bias=False)

        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_2.data.zero_()
        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        #self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=True, eps=1e-8)
        self.batch_norm_2 = BigGANBatchNorm(inputs, condition_vector_dim=256, n_stats=51, eps=1e-12, conditional=True)
        #self.inver_mod2 = ln.Linear(2 * inputs, latent_size)
        if has_second_conv:
            if fused_scale:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 2, 1, bias=False)
            else:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)
        self.fused_scale = fused_scale
        
        self.inputs = inputs
        self.outputs = outputs

        if self.inputs != self.outputs:
            self.batch_norm_3 = BigGANBatchNorm(inputs, condition_vector_dim=256, n_stats=51, eps=1e-12, conditional=True)
            self.conv_3 = ln.Conv2d(inputs, outputs, 1, 1, 0)
            #self.instance_norm_3 = nn.InstanceNorm2d(outputs, affine=True, eps=1e-8)
        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x, cond_vector, truncation=0.4):
        residual = x
        # mean1 = torch.mean(x, dim=[2, 3], keepdim=True) # [b, c, 1, 1]
        # std1 = torch.sqrt(torch.mean((x - mean1) ** 2, dim=[2, 3], keepdim=True))  # [b, c, 1, 1]
        # style1 = torch.cat((mean1, std1), dim=1) # [b,2c,1,1]
        # w1 = self.inver_mod1(style1.view(style1.shape[0],style1.shape[1])) # [b,2c]->[b,512]
        w1=0

        x = self.batch_norm_1(x, truncation, cond_vector)
        #x = F.leaky_relu(x, 0.2)
        x = self.conv_1(x)
        #x = self.instance_norm_1(x)
        x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1, tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]], dtype=torch.float).to(x.device))
        x = x + self.bias_1
        x = F.leaky_relu(x, 0.2)

        # mean2 = torch.mean(x, dim=[2, 3], keepdim=True) # [b, c, 1, 1]
        # std2 = torch.sqrt(torch.mean((x - mean2) ** 2, dim=[2, 3], keepdim=True))  # [b, c, 1, 1]
        # style2 = torch.cat((mean2, std2), dim=1) # [b,2c,1,1]
        # w2 = self.inver_mod2(style2.view(style2.shape[0],style2.shape[1])) # [b,512] , 这里style2.view一直写错成style1.view
        w2=0

        if self.has_second_conv:
            x = self.batch_norm_2(x, truncation, cond_vector)
            #x = F.leaky_relu(x, 0.2)
            x = self.conv_2(x)
            #x = self.instance_norm_2(x)
            x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2, tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]], dtype=torch.float).to(x.device))
            x = x + self.bias_2
            x = F.leaky_relu(x, 0.2)
            if self.inputs != self.outputs: 
                residual = self.batch_norm_3(residual, truncation, cond_vector)
                #x = F.leaky_relu(x, 0.2)
                residual = self.conv_3(residual)
                x = F.leaky_relu(x, 0.2)
            x = x + residual
            if not self.fused_scale: #上采样
                x = F.avg_pool2d(x, 2, 2)

        #x = 0.111*x+0.889*residual #降低x的比例，可以将const的loss缩小！！0.7*residual： 10-11 >> 7 同时 c_s的loss扩大至3， ws的抖动提前, 效果更好
        return x, w1, w2


class BE(nn.Module):
    def __init__(self, startf=16, maxf=512, layer_count=9, latent_size=512, channels=3, pggan=False, biggan=False):
        super().__init__()
        self.maxf = maxf
        self.startf = startf
        self.latent_size = latent_size
        #self.layer_to_resolution = [0 for _ in range(layer_count)]
        self.decode_block = nn.ModuleList()
        self.layer_count = layer_count
        inputs = startf # 16 
        outputs = startf*2
        #resolution = 1024
        # from_RGB = nn.ModuleList()
        fused_scale = False
        self.FromRGB = FromRGB(channels, inputs)

        for i in range(layer_count):

            has_second_conv = i+1 != layer_count #普通的D最后一个块的第二层是 mini_batch_std
            #fused_scale = resolution >= 128 # 在新的一层起初 fused_scale = flase, 完成上采样
            #from_RGB.append(FromRGB(channels, inputs))

            block = BEBlock(inputs, outputs, latent_size, has_second_conv, fused_scale=fused_scale)

            inputs = inputs*2
            outputs = outputs*2
            inputs = min(maxf, inputs) 
            outputs = min(maxf, outputs)
            #self.layer_to_resolution[i] = resolution
            #resolution /=2
            self.decode_block.append(block)
        #self.FromRGB = from_RGB

        self.biggan = biggan
        if biggan:
            self.new_final_1 = ln.Linear(8192, 256, gain=1) # 8192 = 512 * 16
            self.new_final_2 = ln.Linear(256, 128, gain=1)
            #self.new_final_3 = ln.Linear(256, 1000, gain=1) # 

    #将w逆序，以保证和G的w顺序, block_num控制progressive,在其他网络中无效
    def forward(self, x, cond_vector, block_num=9):
        #x = self.FromRGB[9-block_num](x) #每个block一个
        x = self.FromRGB(x)
        #w = torch.tensor(0)
        for i in range(9-block_num,self.layer_count):
            x,w1,w2 = self.decode_block[i](x, cond_vector, truncation=0.4)
            #w_ = torch.cat((w2.view(x.shape[0],1,512),w1.view(x.shape[0],1,512)),dim=1) # [b,2,512]
            # if i == (9-block_num): #最后一层
            #     w = w_ # [b,n,512]
            # else:
            #     w = torch.cat((w_,w),dim=1)
        if self.biggan:
            c_v = self.new_final_1(x.view(x.shape[0],-1)) #[n, 256], cond_vector
            z = self.new_final_2(c_v) # [n, 128]
            #w_ = self.new_final_3(x) # [n, 1000]
        return c_v, z

#test
# E = BE(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3)
# imgs1 = torch.randn(2,3,256,256)
# const2,w2 = E(imgs1)
# print(const2.shape)
# print(w2.shape)
# print(E)
