import matplotlib.pyplot as plt
import jax.numpy as np


class Plotting:
    @staticmethod
    def print_colormap(
        array: np.ndarray,
        title: str = "",
        colormap: str = "viridis",
        colorbar: bool = False,
    ):
        plt.imshow(array, cmap=colormap)
        if colorbar:
            plt.colorbar()
        plt.title(title)
        plt.show()
