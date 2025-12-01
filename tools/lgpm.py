import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class TokenMixer_For_Local(nn.Module):
    """Local feature mixer (spatial domain)
        Function: Use depthwise separable convolutions with different hole rates to capture local spatial features in different ranges
        design
        -Divide the input features into two paths (dim → dim/2+dim/2)
        -Use 3x3 depth convolutions with hole rates of 1 and 2 respectively to expand the receptive field
        Input: x (Tensor) - [B, dim, T, R, A]
        Output: x (Tensor) - [B, dim, T, R, A] (local feature enhancement)
    """
    def __init__(self, dim):
        super(TokenMixer_For_Local, self).__init__()
        self.dim = dim
        self.dim_sp = dim //2 # Number of channels per channel
        # Dilated convolution: expanding the receptive field without increasing parameters
        self.CDilated_1 = nn.Conv3d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv3d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1) # Divided into two channels by channel
        cd1 = self.CDilated_1(x1)  # dilated rate 1: receptive field 3x3
        cd2 = self.CDilated_2(x2)  # dilated rate 2: receptive field 5x5
        x = torch.cat([cd1, cd2], dim=1)  # concatenate back into the dim channel
        return x

class TokenMixer_For_Gloal(nn.Module):
    """Global feature mixer (frequency domain)
        Function: Transform features to the frequency domain through Fourier transform, capture global structural information, and enhance stability through residual connections
        Core: Relying on Fourier Unit to achieve frequency domain feature processing
        Input: x (Tensor) - [B, dim, T, R, A]
        Output: x (Tensor) - [B, dim, T, R, A] (Global Feature Enhancement)
    """
    def __init__(self, dim):
        super(TokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        # 1x1 convolution expands the number of channels to 2 * dim, preparing for frequency domain processing
        self.conv_init = nn.Sequential(nn.Conv3d(dim, dim*2, 1), nn.GELU())
        # 1x1 Convolutional Compression Channel
        self.conv_fina = nn.Sequential(nn.Conv3d(dim*2, dim, 1), nn.GELU())
        # Fourier Unit: Range Frequency Domain Feature Processing
        self.FFC_R = FourierUnit(self.dim*2, self.dim*2)
        # Fourier Unit: Azimuth Frequency Domain Feature Processing
        self.FFC_A = FourierUnit(self.dim*2, self.dim*2)

    def forward(self, x):
        x = self.conv_init(x)  # dim → 2*dim
        x0 = x  # Residual connection backup

        x_a = self.FFC_A(x) # A Frequency domain processing [B, 2dim, T, R, A]
        x_r = self.FFC_R(x.permute(0,1,2,4,3).contiguous())  # R Frequency domain processing [B, 2dim, T, A, R]

        x = self.conv_fina(x_a + x_r + x0)  # Residual connection + Channel Compression
        return x


class FourierUnit(nn.Module):
    """Fourier unit (frequency domain feature enhancement core)
        Function: Convert spatial domain features to frequency domain, adjust frequency domain components through dynamic convolution, and then convert back to spatial domain
        Advantage: The frequency domain is more sensitive to global structures such as texture and edges, making it suitable for image restoration tasks
        Parameters:
        Groups: Volume accumulation of groups (simple task=1, complex task=4, balancing performance and speed)
        Input: x (Tensor) - [B, in_channels, T, R, A]
        Output: x (Tensor) - [B, out_channels, T, R, A] (Global Feature Enhancement)
    """
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.bn = nn.BatchNorm3d(out_channels * 2)  # Normalization of frequency domain features
        # Frequency domain dynamic convolution: adjusting frequency domain components
        self.fdc = nn.Conv3d(in_channels*2, out_channels*2*self.groups, kernel_size=1, 
                             groups=self.groups, bias=True)
        # Dynamic weight generation: assigning weights to grouped convolutions
        self.weight = nn.Sequential(
            nn.Conv3d(in_channels*2, self.groups, kernel_size=1),
            nn.Softmax(dim=1) # Normalize Weights 
        )
        # Frequency domain position encoding: enhancing frequency domain feature expression
        self.fpe = nn.Conv3d(in_channels*2, in_channels*2, kernel_size=3,
                            padding=1, groups=in_channels*2, bias=True)

    def forward(self, x):
        batch, c, t, h, w= x.size()
        # 1. Space domain → frequency domain: 2D Fourier transform (rfft2 only retains the real part, reduces computation)
        ffted = torch.fft.rfft2(x, norm='ortho')  # Output shape [B, c, t, h, w//2+1] (complex number)
        # 2. Complex numbers are split into real and imaginary parts (each [B, c, t, h, w//2+1]) and concatenated into channel dimensions
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)  # [B, c, t, h, w//2+1, 2]
        ffted = rearrange(ffted, 'b c t h w d -> b (c d) t h w').contiguous()  # [B, 2c, t, h, w//2+1]
        # 3. Frequency domain feature processing: normalization + positional encoding (residual)
        ffted = self.bn(ffted)
        ffted = self.fpe(ffted) + ffted  # Frequency domain position encoding residual
        # 4. Dynamic weight adjustment: assigning weights to grouped convolutions
        dy_weight = self.weight(ffted)  # [B, groups, t, h, w//2+1]
        ffted = self.fdc(ffted).view(batch, self.groups, 2*c, t, h, -1)  # [B, groups, 2c, t, h, w//2+1]
        ffted = torch.einsum('ijktml,ijtml->iktml', ffted, dy_weight)  # Weighted by weight (by group) [B, 2c, t, h, w//2+1]
        # 5. Frequency domain → spatial domain: inverse Fourier transform
        ffted = F.gelu(ffted)
        ffted = rearrange(ffted, 'b (c d) t h w -> b c t h w d', d=2).contiguous()  # Restore the dimension of real and imaginary parts [B, c, t, h, w//2+1, 2]
        ffted = torch.view_as_complex(ffted)  # Merge into complex numbers [B, c, t, h, w//2+1]
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')  # Inverse transformation back to the spatial domain [B, c, t, h, w]
        return output

class Mixer3D(nn.Module):
    """Local Global Feature Mixer
        Function: Divide the input features into two paths, perform local spatial perception and global frequency domain modeling respectively, and then fuse them through channel attention fusion
        Core design:
        -Local Branch (TokenMixer_for_Local): Capturing Local Details with Hollow Convolution
        -Global Branch (TokenMixer_for_Gloal): Capturing Global Structure with Fourier Transform
        -Channel Attention (CA): dynamically adjusting the weights of two features to enhance effective information
        Input: x (Tensor) - [B, dim, T, H, W]
        Output: x (Tensor) - [B, dim, T, H, W] (fuses local and global features)`
    """
    def __init__(self, dim, token_mixer_for_local=TokenMixer_For_Local, token_mixer_for_gloal=TokenMixer_For_Gloal):
        super(Mixer3D, self).__init__()
        self.dim = dim
        # Local feature mixer (spatial domain)
        self.mixer_local = token_mixer_for_local(dim=self.dim,)
        # Global feature mixer (frequency domain)
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim,)
        # Channel fusion convolution (compressing 2 * dim channels back to dim)
        self.ca_conv = nn.Sequential(nn.Conv3d(2*dim, dim, 1),)
        # Channel Attention (CA): Generating channel weights through adaptive pooling and convolution
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global average pooling, obtaining[B, 2*dim, 1, 1, 1]
            nn.Conv3d(2*dim, 2*dim//2, kernel_size=1),  # dimensionality reduction
            nn.ReLU(inplace=True),
            nn.Conv3d(2*dim//2, 2*dim, kernel_size=1),  # dimensional lifting
            nn.Sigmoid()  # Generate channel weights of 0-1
        )
        self.gelu = nn.GELU()  # Activation function, introducing nonlinearity
        # 1x1 convolution: Expand the input dim channel to 2 * dim for splitting
        self.conv_init = nn.Sequential(nn.Conv3d(dim, 2*dim, 1),)

    def forward(self, x):
        # B, T, C, R, A= x.shape
        
        x = x.permute(0, 2, 1, 3, 4).contiguous() # [4, 16, 8, 128, 128] - > [4, 8, 16, 128, 128]
        
        x = self.conv_init(x)  # Number of channels: dim → 2 * dim

        x = list(torch.split(x, self.dim, dim=1))  # Divided into two channels by channel: [x0, x1] (each dim channel)
        x_local = self.mixer_local(x[0])  # Local branch processing x0 [4, 8, 16, 128, 128]
        x_gloal = self.mixer_gloal(x[1])  # Global branch processing x1 [4, 8, 16, 128, 128]
        x = torch.cat([x_local, x_gloal], dim=1)  # concatenate two channel features: 2 * dim channels [4, 16, 16, 128, 128]
        x = self.gelu(x)
        x = self.ca(x) * x  # Channel attention weighting (channel wise multiplication) [4, 16, 16, 1, 1] * [4, 16, 16, 128, 128]
        x = self.ca_conv(x)  # Compress the number of channels back to dim [4, 8, 16, 128, 128]

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x
    