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
        self.synapses_track = []
        self.synapses_net = []
        self.tree = {}
        self.root = root
        self.final = final
        self.target_nodes = []
        self.max_weight = np.max(weight_matrix)
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
                    self.synapses_net.append(self.network.createSynapse(self.neurons[neighbor], self.neurons[node], w=1, d=delay))
                    self.network.createSynapse(self.neurons[neighbor], tracking_neuron, w=1, d=delay)
                    self.synapses_track.append(self.network.createSynapse(self.neurons[neighbor], tracking_neuron, w=1, d=delay))


                    print("neighbor =", neighbor, "node =", node, "delay =", delay)
                    
                    visited[neighbor] = True
                    queue.append(neighbor)

        print("nr. synapses to track =", len(self.synapses_track), "nr. synapses in net =", len(self.synapses_net))

        return dict(tree)
    
    def construct(self):
        self.network = Network()
        counter_neurons = 3 # to id all neurons, 1 and 2 are for tracking- and start neurons
        # create for every node in the input graph a neuron:
        for i in range(self.nr_nodes):
            neuron = self.network.createLIF(m=0, V_init=0, V_reset=0, thr=1, read_out=False, ID=counter_neurons)
            self.neurons.append(neuron)
            counter_neurons += 1
            
        # If weights are not unique, make them so
        # first check if weights are unique:
        import collections
        weights = [self.weight_matrix[i][j] for i in range(self.nr_nodes) for j in range(self.nr_nodes) if self.weight_matrix[i][j] != 0]
        unique_weights = list(collections.Counter(weights))

        if len(unique_weights) < len(weights):
            # no unique weights, must perform the mapping:
            counter = 1
            self.weight_matrix_transformed = np.zeros((self.nr_nodes, self.nr_nodes))
            #weight_matrix_transformed = [[0 for i in range(self.nr_nodes)] for j in range(self.nr_nodes)]
            unique_weights, weight_counts = np.unique(self.weight_matrix, return_counts=True)
            max_duplicate_weights = weight_counts[1] if 0 in unique_weights else weight_counts[0]

            weight_history = []
            for i in range(self.nr_nodes):
                for j in range(i+1, self.nr_nodes):
                    w = self.weight_matrix[i,j]
                    if i != j and w > 0:
                        weight_count = np.count_nonzero(weight_history == w)
                        weight = int(float(max_duplicate_weights * w + weight_count))
                        weight_history.append(w)
                        self.weight_matrix_transformed[i,j] = weight
                        self.weight_matrix_transformed[i,j] = int(float(self.weight_matrix_transformed[i,j]))

                        # Increase counter
                        counter+=1

                        # Keep track of maximum delay
                        if weight > self.max_weight:
                            self.max_weight = weight
            
            self.weight_matrix_transformed = self.weight_matrix_transformed.astype(int) # to convert float.64 entries to integers
        
        else:
            # if weights are already unique: assign the orignal matrix to the unique matrix
            self.weight_matrix_transformed = self.weight_matrix
        
        print("Unique weight matrix =", self.weight_matrix_transformed)

         # Calling build_tree builds the tree and creates the synapses between the neurons just created
        self.tree = self.build_tree(self.weight_matrix_transformed)
        print("Tree =", self.tree)
        # the input train starts the interation and has to be connected to the desired final neuron:
        start_train = self.network.createInputTrain([1], loop=False, ID=2)
        self.network.createSynapse(start_train, self.neurons[self.final], w=1, d=1)
        #print("self.final =", self.final)

        return self.network



def construct_widestpath(spikes, root, weight_matrix):
    cycle_starts = np.argwhere(spikes[:, 0] == True).flatten()
    # all subsequent delays in the temporal output pattern are:
    delay_list = [cycle_starts[0]-1] + [cycle_starts[i]-cycle_starts[i-1] for i in range(1,len(cycle_starts))]
    print("Sequence of delays =", delay_list)
    capacity = np.min(delay_list) # capacity is the minimum "weight" which was translated into delays; the bottleneck in the path


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
    nr_steps = len(delay_list) # In order to ensure the path is the correct length
    path = [root]
    print("root initialized =", root)

    for delay in delay_list[::-1]: # all delays encountered from root to final
        root = path[-1] # initializing the root as the last entry of the path
        children = tree[root] # this root bifurcates into its children, given by the tree
        print("Delay =", delay, "root=", root, "children =", children)
        for child in children: # iterating over these chilren and only the child with corresponding delay is chosen
            print("child =", child, "with delay =", weight_matrix[root,child], weight_matrix[child,root])
            if max(weight_matrix[root,child], weight_matrix[child,root]) == delay:
                 path.append(child) # this child is added to the path, in the next iteration this is chosen as new root
        

    if len(path) != nr_steps + 1:
        print("Path is incorrect")

    return capacity, path


def execute_simulator(weight_matrix, root, final):
    constructor = Constructor(weight_matrix, root, final)
    network = constructor.construct()
    unique_weights = constructor.weight_matrix_transformed

    sim = Simulator(network)
    
    # Add all read_out neurons to the simulation
    sim.raster.addTarget(constructor.target_nodes)
    sim.multimeter.addTarget(constructor.target_nodes)
    sim.run(steps=constructor.total_weight+10, plotting=False)

    spikes = sim.raster.get_measurements()
    print("nr of spikes:", np.count_nonzero(spikes))

    return construct_widestpath(spikes, root, unique_weights)


parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str)
args = parser.parse_args()
if args.i is not None:
    weight_matrix = np.array(ast.literal_eval(args.i))
else:
    weight_matrix = np.array([[ 0,  0,  5,  0,  8],
                              [ 0,  0,  0,  8,  0],
                              [ 0,  0,  0,  0,  0],
                              [ 0,  0,  0,  0,  6],
                              [ 0,  0,  0,  0,  0]])
    
#Usage eg python3 createMaST.py -i [[0,1,5,0,9],[0,0,2,7,0],[0,0,0,3,1],[0,0,0,0,14],[0,0,0,0,0]]
# or just python3 createMaST.py
print(f"Graph:\n{weight_matrix}\n")
weight_matrix *= 2
cap, path = execute_simulator(weight_matrix, root=0, final=1)

print(f"Capacity:\n{cap}")
print(f"Path:\n{path}")
