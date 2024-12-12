import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
from core.detectors import Raster, Multimeter


class Simulator:
    """Simulator

    Parameters
    ----------
    network : Network
        Network to simulate
    detectors : List
        List of detectors
    """

    def __init__(self, network, seed=None):
        self.network = network
        self.multimeter = Multimeter()
        self.raster = Raster()
        if seed != None:
            self.network.update_rng(np.random.RandomState(seed))

    def run(self, steps, plotting=False):
        """Run the simulator

        Parameters
        ----------
        steps : int
            Number of steps to simulate
        """
        self.raster.initialize(steps)
        self.multimeter.initialize(steps)

        for _ in range(steps):
            self.network.step()
            self.raster.step()
            self.multimeter.step()

        if plotting:
            self.print_detectors(steps)

    def to_inet_string(self):
        inet_str = ""
        inet_str += self.raster.to_inet_string() + "\n\n"
        inet_str += self.multimeter.to_inet_string() + "\n\n"
        return inet_str

    def print_detectors(self, steps=0):
        rasterdata = self.raster.get_measurements()
        print("Rasterdata:")
        print(rasterdata.T)
        multimeterdata = self.multimeter.get_measurements()
        print("\nMultimeterdata:")
        print(multimeterdata.T)

        ntd = len(rasterdata.T)
        nvd = len(multimeterdata.T)
        overhead = 2 if ntd else 1
        fig, _ = plt.subplots(
            constrained_layout=True, nrows=overhead + nvd, figsize=(7, 7)
        )

        options = {
            "with_labels": True,
            "node_color": "white",
            "edgecolors": "blue",
            "ax": fig.axes[0],
            "node_size": 1100,
            "pos": nx.circular_layout(self.network.graph),
        }
        nx.draw_networkx(self.network.graph, **options)

        if ntd:
            fig.axes[1].matshow(rasterdata.T, cmap="gray", aspect="auto")
            fig.axes[1].set_xticks(np.arange(-0.5, steps, 1), minor=True)
            fig.axes[1].set_yticks(np.arange(-0.5, ntd, 1), minor=True)
            fig.axes[1].grid(which="minor", color="gray", linestyle="-", linewidth=2)
            fig.axes[1].xaxis.set_major_locator(ticker.MultipleLocator(1))
            fig.axes[1].set_yticks(np.arange(0, ntd, 1))
            fig.axes[1].set_yticklabels([t.ID for t in self.raster.targets])

        for i in range(nvd):
            fig.axes[i + overhead].plot(multimeterdata[:, i])
            fig.axes[i + overhead].set_ylim(top=(max(multimeterdata.T[i]) + 0.5))
            fig.axes[i + overhead].set_ylabel(self.multimeter.targets[i].ID)
            fig.axes[i + overhead].grid(visible=None, which="major")
            fig.axes[i + overhead].xaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
