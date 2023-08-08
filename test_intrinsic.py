import torch

H = 375
W = 1242
K = torch.Tensor([[7.215377e+02, 0.000000e+00, 6.095593e+02, 0.0],
                  [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.0],
                  [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])

x_offset = 250
y_offset = 46
K_crop = K.clone().detach()
K_crop[0, 2] = K_crop[0, 2] - x_offset
K_crop[1, 2] = K_crop[1, 2] - y_offset

K_crop_18 = K_crop * 0.125
K_crop_18[2, 2] = 1
K_crop_18[3, 3] = 1

print(K)
print(K_crop)
print(K_crop_18)

uv = torch.tensor([[120.0], [0.0], [1.0]])
xy = torch.linalg.inv(K_crop_18)[:3, :3] @ uv

print(torch.linalg.inv(K_crop_18)[:3, :3])
uv_again = K_crop[:3, :3] @ xy
print(uv_again)
