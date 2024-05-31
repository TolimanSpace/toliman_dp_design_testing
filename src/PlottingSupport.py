import matplotlib.pyplot as plt
import jax.numpy as np


class Plotting:
    @staticmethod
    def printColormap(
        array: np.ndarray,
        title: str = "",
        colormap: str = "inferno",
        colorbar: bool = False,
        colorbar_label: str = "",
    ):
        image = plt.imshow(array, cmap=colormap)
        if colorbar:
            clbr = plt.colorbar(image)
            clbr.set_label(colorbar_label)
        plt.title(title)
        plt.show()
