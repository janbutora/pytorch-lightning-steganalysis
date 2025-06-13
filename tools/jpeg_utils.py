import numpy as np
from skimage.util import view_as_blocks

cc, rr = np.meshgrid(np.arange(8), np.arange(8))
T = np.sqrt(2 / 8) * np.cos(np.pi * (2 * cc + 1) * rr / (2 * 8))
T[0, :] /= np.sqrt(2)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float32)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    return rgb

def decompress_view(C, Q):
    return (T.T) @ (C * Q) @ (T) + 128

def decompress_image(C, Q):
    view = decompress_view(view_as_blocks(C, (8, 8)), Q)
    I = reshape_view_to_original(view, C)
    return I

def reshape_view_to_original(arr, orig):
    return np.transpose(
        arr.reshape(orig.shape[0] // 8, orig.shape[1] // 8, 8, 8), [0, 2, 1, 3]
    ).reshape(orig.shape)
