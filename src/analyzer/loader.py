import numpy as np
from scipy.io import loadmat
import numpy.typing as npt


class MatLoader:
    def __init__(self, file):
        mat_images: npt.NDArray = loadmat(file)["data"]
        images = mat_images.swapaxes(0, 2)
        self.images = images.swapaxes(1, 2)

class NpzLoader:
    def __init__(self, file: str):
        self.data =  dict(np.load(file))


if __name__ == "__main__":
    pass
