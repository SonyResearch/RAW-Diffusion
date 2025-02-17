import torch

def rggb_to_rgb(data):
    assert data.shape[1] == 4
    img_r = data[:, 0]
    img_g1 = data[:, 1]
    img_g2 = data[:, 2]
    img_b = data[:, 3]

    img_g = (img_g1 + img_g2) / 2

    img = torch.stack((img_r, img_g, img_b), axis=1)
    return img

