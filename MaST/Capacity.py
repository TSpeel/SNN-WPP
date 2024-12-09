# create new SNNs to find path

import numpy as np
from core.networks import Network
from core.simulators import Simulator
import argparse
import ast

class Constructor():
    """Constructor

    Parameters
    ----------
    weight_matrix : list
        Matrix representation of network, where connections are represented as non-zero elements
    """
    def __init__(self, weight_matrix: np.ndarray, root, final) -> None:
        self.weight_matrix = weight_matrix
        self.nr_nodes, _ = weight_matrix.shape
        self.nr_edges = np.count_nonzero(weight_matrix)
        self.neurons = []
        self.tree = {}
        self.root = root
        self.final = final
        self.target_nodes = []
        #self.max_weight = np.max(weight_matrix)
        self.total_weight = np.sum(weight_matrix)


    def build_tree(self, adj_matrix):
        """
        Build a tree from a half adjacency matrix, given a root node. Returns the tree as dictionary.
        """
        from collections import defaultdict, deque

        n = len(adj_matrix)  # Number of nodes
        tree = defaultdict(list)  # To store the tree structure
        visited = [False] * n     # Track visited nodes

        # read out neuron whose temporal output pattern indicates the neuron-path:
        tracking_neuron = self.network.createLIF(m=0, V_init=0, V_reset=0, thr=1, read_out=True, ID=1) 
        self.target_nodes.append(tracking_neuron)

        # Build a full adjacency list from the half adjacency matrix
        adjacency_list = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):  # Only consider the upper triangular part (i < j)
                if adj_matrix[i][j] > 0:
                    adjacency_list[i].append(j)
                    adjacency_list[j].append(i)


        queue = deque([self.root])
        visited[self.root] = True
        print("Destination neuron =", self.final, "Root neuron =", self.root)    

        while queue:
            node = queue.popleft()

            for neighbor in adjacency_list[node]:
                if not visited[neighbor]:
                    tree[node].append(neighbor)
                    delay = max(adj_matrix[node][neighbor], adj_matrix[neighbor][node]) # delay is "weight" of connection
                    # create synapses; from one node (neighbor) to its previous node (node) and to the tracking neuron:
                    self.network.createSynapse(self.neurons[neighbor], self.neurons[node], w=1, d=delay)
                    self.network.createSynapse(self.neurons[neighbor], tracking_neuron, w=1, d=delay)

                    #print("neighbor =", neighbor, "node =", node, "delay =", delay)
                    
                    visited[neighbor] = True
                    queue.append(neighbor)

        return dict(tree)



    def construct(self):
        self.network = Network()
        counter_neurons = 3 # to id all neurons, 1 and 2 are for tracking- and start neurons
        # create for every node in the input graph a neuron:
        for i in range(self.nr_nodes):
            neuron = self.network.createLIF(m=0, V_init=0, V_reset=0, thr=1, read_out=False, ID=counter_neurons)
            self.neurons.append(neuron)

        # Calling build_tree builds the tree and creates the synapses between the neurons just created
        self.tree = self.build_tree(weight_matrix)
        print("Tree =", self.tree)
        # the input train starts the interation and has to be connected to the desired final neuron:
        start_train = self.network.createInputTrain([1], loop=False, ID=2)
        self.network.createSynapse(start_train, self.neurons[self.final], w=1, d=1)

        return self.network



def construct_widestpath(spikes, root, weight_matrix):
    cycle_starts = np.argwhere(spikes[:, 0] == True).flatten()
    # all subsequent delays in the temporal output pattern are:
    delay_list = [cycle_starts[0]-1] + [cycle_starts[i]-cycle_starts[i-1] for i in range(1,len(cycle_starts))]
    print("Delays from temporal output tracking neuron:", delay_list)

    capacity = np.min(delay_list) # capacity is the minimum "weight" which was translated into delays; the bottleneck in the path
    #total_path_time = np.sum(delay_list) # the number of time-steps

    # construct tree manually again because we cannot call self in this function...
    # it is copy paste from the previous tree function, but without creating synapses
    from collections import defaultdict, deque

    n = len(weight_matrix)  # Number of nodes
    tree = defaultdict(list)  # To store the tree structure
    visited = [False] * n     # Track visited nodes

    adjacency_list = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):  # Only consider the upper triangular part (i < j)
            if weight_matrix[i][j] > 0:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    queue = deque([root])
    visited[root] = True

    while queue:
        node = queue.popleft()

        for neighbor in adjacency_list[node]:
            if not visited[neighbor]:  # Unvisited node
                tree[node].append(neighbor)
                delay = max(weight_matrix[node][neighbor], weight_matrix[neighbor][node])
                visited[neighbor] = True
                queue.append(neighbor)

    # the tree is created, now use this tree to construct the path: which neurons are visited
    count = 0
    path = [root] # path starts at the root

    for delay in delay_list[::-1]: # all delays encountered from root to final
        root = path[-1] # initializing the root as the last entry of the path
        children = tree[root] # this root bifurcates into its children, given by the tree

        for child in children: # iterating over these chilren and only the child with corresponding delay is chosen
            if max(weight_matrix[root,child], weight_matrix[child,root]) == delay:
                 path.append(child) # this child is added to the path, in the next iteration this is chosen as new root

    spikes_at = [np.flatnonzero(spikes[:, node])[0] if spikes[:, node].any() else 99999 for node in range(3, spikes.shape[1])]

    return capacity, path


def execute_simulator(weight_matrix, root, final):
    constructor = Constructor(weight_matrix, root, final)
    network = constructor.construct()
    weights = weight_matrix
    sim = Simulator(network)
    
    # Add all read_out neurons to the simulation
    sim.raster.addTarget(constructor.target_nodes)
    sim.multimeter.addTarget(constructor.target_nodes)
    sim.run(steps=constructor.total_weight+10, plotting=False)

    spikes = sim.raster.get_measurements()

    return construct_widestpath(spikes, root, weight_matrix)


parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str)
args = parser.parse_args()
if args.i is not None:
    weight_matrix = np.array(ast.literal_eval(args.i))
else:
    weight_matrix = np.array([[ 0,  0,  5,  0,  9],
                      [ 0,  0,  0,  10,  0],
                      [ 0,  0,  0,  0,  0],
                      [ 0,  0,  0,  0,  2],
                      [ 0,  0,  0,  0,  0]])
    
#Usage eg python3 createMaST.py -i [[0,1,5,0,9],[0,0,2,7,0],[0,0,0,3,1],[0,0,0,0,14],[0,0,0,0,0]]
# or just python3 createMaST.py
print(f"Graph:\n{weight_matrix}\n")
weight_matrix *= 2
cap, path = execute_simulator(weight_matrix, root=0, final=1)

print(f"Capacity:\n{cap}")
print(f"Path:\n{path}")