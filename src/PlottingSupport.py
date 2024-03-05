import matplotlib.pyplot as plt
import jax.numpy as np


class Plotting:
    @staticmethod
    def printColormap(
        array: np.ndarray,
        title: str = "",
        colormap: str = "inferno",
        colorbar: bool = False,
    ):
        plt.imshow(array, cmap=colormap)
        if colorbar:
            plt.colorbar()
        plt.title(title)
        plt.show()
