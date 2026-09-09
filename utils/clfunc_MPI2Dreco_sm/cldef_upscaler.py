import torch
import torch.nn.functional as F


class UpscaleOperator:
    # upsampling operator U for (N,C,H,W) images with integer scale s
    # mode='repeat' -> nearest neighbor block duplication
    # mode='bilinear' / 'bicubic' -> F.interpolate
    #
    # U(x)    forward upsample
    # UT(z)   adjoint (true transpose) of U
    # UTU(x)  U^T U via autograd, works for any mode
    # UTU2(x) closed form U^T U, only for repeat/bilinear (faster, no autograd)

    def __init__(self, s, mode='bilinear', align_corners=False):
        if not (isinstance(s, int) and s >= 1):
            raise ValueError("s must be a positive integer.")
        if mode not in ('repeat', 'bilinear', 'bicubic'):
            raise ValueError("mode must be 'repeat', 'bilinear', or 'bicubic'.")
        self.s = s
        self.mode = mode
        self.align_corners = align_corners

    def U(self, x):
        # x: (N,C,H,W) low-res -> (N,C,H*s,W*s) high-res
        if x.ndim != 4:
            raise ValueError("Expected x with shape (N, C, H, W).")

        if self.mode == 'repeat':
            y = x.repeat_interleave(self.s, dim=-2).repeat_interleave(self.s, dim=-1)
        else:
            y = F.interpolate(x, scale_factor=self.s, mode=self.mode,
                               align_corners=self.align_corners, recompute_scale_factor=False)

        return y.squeeze()

    def UT(self, z, out_hw=None):
        # z: (N,C,Hh,Wh) high-res -> (N,C,H,W) low-res, i.e. U^T z
        if z.ndim != 4:
            raise ValueError("Expected z with shape (N, C, Hh, Wh).")

        if out_hw is None:
            H_lr = z.shape[-2] // self.s
            W_lr = z.shape[-1] // self.s
        else:
            H_lr, W_lr = out_hw

        if self.mode == 'repeat':
            # adjoint of duplication is just block-sum pooling
            if z.shape[-2] != H_lr * self.s or z.shape[-1] != W_lr * self.s:
                raise ValueError("HR size must be exactly (H * s, W * s) for 'repeat' mode.")
            v = F.avg_pool2d(z, kernel_size=self.s, stride=self.s) * (self.s * self.s)
            return v.squeeze()

        # for bilinear/bicubic just get the adjoint via VJP through interpolate
        x_dummy = torch.zeros((z.shape[0], z.shape[1], H_lr, W_lr),
                               dtype=z.dtype, device=z.device, requires_grad=True)
        y = F.interpolate(x_dummy, size=(H_lr * self.s, W_lr * self.s),
                           mode=self.mode, align_corners=self.align_corners)
        vjp, = torch.autograd.grad(y, x_dummy, grad_outputs=z)

        return vjp.squeeze()

    def UTU(self, x):
        # generic U^T U, works for all modes since it just chains U and UT
        if x.ndim != 4:
            raise ValueError("Expected x with shape (N, C, H, W).")

        y = self.U(x)
        # U() squeezes singleton dims away when N=C=1, put them back before UT
        if y.ndim == 2:
            y = y.unsqueeze(0).unsqueeze(0)
        elif y.ndim == 3:
            y = y.unsqueeze(0)

        v = self.UT(y, out_hw=(x.shape[-2], x.shape[-1]))
        return v.squeeze()

    def UTU2(self, x, padding='reflect'):
        # fast closed-form U^T U, no autograd needed
        # repeat -> s^2 * I, bilinear -> separable 3x3 kernel, bicubic -> not implemented (use UTU)
        if x.ndim != 4:
            raise ValueError("Expected x with shape (N,C,H,W).")

        if self.mode == 'repeat':
            return (self.s * self.s) * x

        if self.mode == 'bicubic':
            raise NotImplementedError(
                "Closed-form UTU2 kernel is only implemented for 'bilinear'. Use UTU(x) for 'bicubic' mode."
            )

        s = self.s
        c0 = (2 * s * s + 1) / (3.0 * s)
        c1 = (s * s - 1) / (6.0 * s)

        N, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        if padding not in ('reflect', 'replicate'):
            raise ValueError("padding must be 'reflect' or 'replicate'.")

        # depthwise separable conv: horizontal pass then vertical pass
        k_horiz = torch.tensor([c1, c0, c1], dtype=dtype, device=device).view(1, 1, 1, 3).repeat(C, 1, 1, 1)
        x_pad = F.pad(x, (1, 1, 0, 0), mode=padding)
        y = F.conv2d(x_pad, k_horiz, stride=1, padding=0, groups=C)

        k_vert = torch.tensor([c1, c0, c1], dtype=dtype, device=device).view(1, 1, 3, 1).repeat(C, 1, 1, 1)
        y_pad = F.pad(y, (0, 0, 1, 1), mode=padding)
        out = F.conv2d(y_pad, k_vert, stride=1, padding=0, groups=C)

        return out