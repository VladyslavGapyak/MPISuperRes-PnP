import torch.nn as nn

class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4):
        super(UNetRes, self).__init__()

        #Preparation
        downsample_blocks = []
        upsample_blocks = []
        for i in range(3):
            downsample_blocks.append(nn.Conv2d(in_channels=nc[i], out_channels=nc[i+1], kernel_size=2, stride=2, padding=0, bias=False))
            upsample_blocks.append(nn.ConvTranspose2d(in_channels=nc[i+1], out_channels=nc[i], kernel_size=2, stride=2, padding=0, bias=False))
        
        #Building blocks of the residual UNet
        self.m_head = nn.Conv2d(in_channels=in_nc, out_channels=nc[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        self.m_down1 = nn.Sequential(*[ResBlock(nc[0], nc[0], bias=False) for _ in range(nb)], downsample_blocks[0])
        self.m_down2 = nn.Sequential(*[ResBlock(nc[1], nc[1], bias=False) for _ in range(nb)], downsample_blocks[1])
        self.m_down3 = nn.Sequential(*[ResBlock(nc[2], nc[2], bias=False) for _ in range(nb)], downsample_blocks[2])

        self.m_body  = nn.Sequential(*[ResBlock(nc[3], nc[3], bias=False) for _ in range(nb)])

        self.m_up3 = nn.Sequential(upsample_blocks[2], *[ResBlock(nc[2], nc[2], bias=False) for _ in range(nb)])
        self.m_up2 = nn.Sequential(upsample_blocks[1], *[ResBlock(nc[1], nc[1], bias=False) for _ in range(nb)])
        self.m_up1 = nn.Sequential(upsample_blocks[0], *[ResBlock(nc[0], nc[0], bias=False) for _ in range(nb)])

        self.m_tail = nn.Conv2d(in_channels=nc[0], out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        return x



class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResBlock, self).__init__()

        L = []
        L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        
        self.res = nn.Sequential(*L)
    
    def forward(self, x):
        return x + self.res(x)

    
