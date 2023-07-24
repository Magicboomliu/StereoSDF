import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SDF_MLP(nn.Module):
    def __init__(self, feat_length, cv_feat_length, multires=0,
                 d_hidden=128, n_layers=4, skip_in=(), geometric_init=True) -> None:
        super().__init__()

        if multires == 0:
            d_in = feat_length * 2 + cv_feat_length + 3
            self.embed_fn_fine = None
        else:
            raise NotImplementedError('Positional encoding is not implemented yet.')
        
        d_out = 1
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.num_layers = len(dims)
        self.skip_in = skip_in
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            # Refer to Neus git
            inside_outside = False  # hard coded
            bias = 0.5  # hard coded
            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            
            weight_norm = True  # hard coded
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, 'lin'+str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, feat_left, feat_right, cv_left, xyz):
        """
        feat_left: (N, F)
        feat_right: (N, F)
        cv_left: (N, D)
        xyz: (N, 3)

        return: (N, 1)
        """
        if self.embed_fn_fine is not None:
            xyz = self.embed_fn_fine(xyz)

        try:
            inputs = torch.cat([feat_left, feat_right, cv_left, xyz], dim=-1)
        except:
            print(feat_left.shape)
            print(feat_right.shape)
            print(cv_left.shape)
            print(xyz.shape)
            exit()
        
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

    def gradient(self, feat_left, feat_right, cv_left, xyz):
        xyz.requires_grad_(True)
        y = self.forward(feat_left, feat_right, cv_left, xyz)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=xyz,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)

        return gradients[0].unsqueeze(1)


if __name__ == '__main__':
    feat_length = 512
    cv_feat_length = 128

    sdf_mlp = SDF_MLP(feat_length=feat_length, cv_feat_length=cv_feat_length)
    feat_left = torch.randn(512, 512)
    feat_right = torch.randn(512, 512)
    cv_left = torch.randn(512, 128)
    xyz = torch.randn(512, 3)

    x = sdf_mlp(feat_left, feat_right, cv_left, xyz)
    print(x.shape)
    gradient = sdf_mlp.gradient(feat_left, feat_right, cv_left, xyz)
    print(gradient.shape)
