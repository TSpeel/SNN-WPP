# create new SNNs to find path

import numpy as np
from core.networks import Network
from core.simulators import Simulator
import argparse
import ast
import sys


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
        self.max_weight = np.max(weight_matrix)
        self.total_weight = np.sum(weight_matrix)
    

    def build_tree(self, adj_matrix):
        """
        Build a tree from a half adjacency matrix, given a root node. Returns the tree as dictionary. Already constructs synapses.
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

        while queue:
            node = queue.popleft()
            for neighbor in adjacency_list[node]:
                if not visited[neighbor]:
                    tree[node].append(neighbor)
                    delay = max(adj_matrix[node][neighbor], adj_matrix[neighbor][node]) # delay is "weight" of connection
                   
                    # create synapses; from one node (neighbor) to its previous node (node) and to the tracking neuron:
                    self.network.createSynapse(self.neurons[neighbor], self.neurons[node], w=1, d=delay)
                    self.network.createSynapse(self.neurons[neighbor], tracking_neuron, w=1, d=delay)

                    visited[neighbor] = True
                    queue.append(neighbor)

        return dict(tree)


    def construct(self):
        """ Construct the network, create the neurons. If duplicate weights are present, map them to unique weights.
         Call the tree function to construct a tree with the desired initial vertex as root neuron. The tree function creates the synapses """
        
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
            counter = 1 # not used, why here?
            self.weight_matrix_transformed = np.zeros((self.nr_nodes, self.nr_nodes))
            occurence = collections.Counter(weights).most_common(1)[0][1]

            weight_history = []
            for i in range(self.nr_nodes):
                for j in range(i+1, self.nr_nodes):
                    w = self.weight_matrix[i,j]
                    if i != j and w > 0:
                        weight_count = np.count_nonzero(weight_history == w)
                        weight = int(float(w * occurence + weight_count))
                        weight_history.append(w)
                        self.weight_matrix_transformed[i,j] = weight
                        self.weight_matrix_transformed[i,j] = int(float(self.weight_matrix_transformed[i,j]))

                        # Increase counter
                        counter+=1

                        # Keep track of maximum delay
                        if weight > self.max_weight:
                            self.max_weight = weight
                        
            
            self.weight_matrix_transformed = self.weight_matrix_transformed.astype(int) # to convert float.64 entries to integers
            self.total_weight = np.sum(self.weight_matrix_transformed)
        
        else:
            # if weights are already unique: assign the orignal matrix to the unique matrix
            self.weight_matrix_transformed = self.weight_matrix        
        #print("uniques", self.weight_matrix_transformed)

        # test that the matrix is indeed unique
        all_un_weights = [self.weight_matrix_transformed[i][j] for i in range(self.nr_nodes) for j in range(self.nr_nodes) if self.weight_matrix_transformed[i][j] != 0]
        actual_un_weights = list(collections.Counter(all_un_weights))
        if len(actual_un_weights) < len(all_un_weights):
            print("Weights are not yet unique")
            sys.exit()

         # Calling build_tree builds the tree and creates the synapses between the neurons just created
        self.tree = self.build_tree(self.weight_matrix_transformed)

        # the input train starts the interation and has to be connected to the desired final neuron:
        start_train = self.network.createInputTrain([1], loop=False, ID=2)
        self.network.createSynapse(start_train, self.neurons[self.final], w=1, d=1)

        return self.network



def construct_widestpath(spikes, root, weight_matrix, og_matrix):
    cycle_starts = np.argwhere(spikes[:, 0] == True).flatten()
    # all subsequent delays in the temporal output pattern are:
    delay_list = [cycle_starts[0]-1] + [cycle_starts[i]-cycle_starts[i-1] for i in range(1,len(cycle_starts))]
    capacity = np.min(delay_list) # capacity is the minimum "weight" which was translated into delays; the bottleneck in the path

    # to obtain the correct capacity we must reverse the pre-processing step
    if np.array_equal(weight_matrix, og_matrix) == False: # only if weights are mapped, thus are nog their og values
        # the indices of the elements are unchanged
        i,j = np.where(weight_matrix == capacity)
        capacity = og_matrix[i[0]][j[0]]

    else: capacity = capacity

    # construct tree manually again because we cannot call self in this function
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
    
    path = [root]

    for delay in delay_list[::-1]: # all delays encountered from root to final
        root = path[-1] # initializing the root as the last entry of the path
        children = tree[root] # this root bifurcates into its children, given by the tree
        for child in children: # iterating over these chilren and only the child with corresponding delay is chosen
            if max(weight_matrix[root,child], weight_matrix[child,root]) == delay:
                 path.append(child) # this child is added to the path, in the next iteration this is chosen as new root
    
    # To ensure the path is the correct length
    nr_steps = len(delay_list) 
    if len(path) != nr_steps + 1:
        print("Path is not the correct number of nodes")
        sys.exit()

    return capacity, path


def execute_simulator(weight_matrix, root, final):
    # immediately check if the input is indeed a tree
    # 1: def of tree includes that the nr of vertices = nr of edges + 1. Otherwise there are cycles or unconnected vertices
    vertices = weight_matrix.shape[0]
    edges = np.count_nonzero(weight_matrix)
    if vertices != edges + 1:
        print("The input graph has a cycle and is hence not a tree")
        sys.exit()
    
    # 2: a tree cannot have unconnected nodes
    for i in range(len(weight_matrix)): # iterate over the rows
        if np.all(weight_matrix[i] == 0): # if a row is fully empty
            if np.all(weight_matrix[:,i] == 0): # check if the equivalent column is empty as well
                print("Unconnected node") # in that case, this vertex is unconnected
                sys.exit() # once this is detected, stop the script
    
    # the input cannot have negative weights:
    for i in range(len(weight_matrix)):
        for j in range(len(weight_matrix)):
            if weight_matrix[i][j] < 0:
                print("Negative weight in input")
                sys.exit()

    constructor = Constructor(weight_matrix, root, final)
    network = constructor.construct()
    unique_weights = constructor.weight_matrix_transformed
    original_weigths = weight_matrix

    sim = Simulator(network)
    
    # Add all read_out neurons to the simulation
    sim.raster.addTarget(constructor.target_nodes)
    sim.multimeter.addTarget(constructor.target_nodes)
    sim.run(steps=constructor.total_weight+10, plotting=False)

    spikes = sim.raster.get_measurements()

    return construct_widestpath(spikes, root, unique_weights, original_weigths)


parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str)
args = parser.parse_args()
if args.i is not None:
    weight_matrix = np.array(ast.literal_eval(args.i))
else:
    weight_matrix = np.array([[ 0,  9,  10,  0,  7],
                              [ 0,  0,  0,  0,  0],
                              [ 0,  0,  0,  2,  0],
                              [ 0,  0,  0,  0,  0],
                              [ 0,  0,  0,  0,  0]])
    

#Usage eg python3 createMaST.py -i [[0,1,5,0,9],[0,0,2,7,0],[0,0,0,3,1],[0,0,0,0,14],[0,0,0,0,0]]
# or just python3 createMaST.py

#print(f"Graph:\n{weight_matrix}\n")

def edge_cases():
    duplicates = np.array([[ 0,  9,  10,  0,  9],
                              [ 0,  0,  0,  0,  0],
                              [ 0,  0,  0,  2,  0],
                              [ 0,  0,  0,  0,  0],
                              [ 0,  0,  0,  0,  0]])
    
    uniques =       np.array([[ 0,  18,  20,  0,  19],
                              [ 0,  0,  0,  0,  0],
                              [ 0,  0,  0,  4,  0],
                              [ 0,  0,  0,  0,  0],
                              [ 0,  0,  0,  0,  0]])
    
    all_duplicate = np.array([[ 0,  5,  5,  0,  0],
                              [ 0,  0,  0,  0,  0],
                              [ 0,  0,  0,  5,  0],
                              [ 0,  0,  0,  0,  5],
                              [ 0,  0,  0,  0,  0]])

    big_matrix = np.array([[ 0,  18,  6,  0,  0,  0,  0,  3,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  2,  20,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  9,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0,  6,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0,  15, 0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  8],
                        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]) # with duplicate weight!


    # these should yield the same path:
    if execute_simulator(duplicates, root=0, final=3)[1] == execute_simulator(uniques, root=0, final=3)[1]:
        print("Path given is equivalent for weight-mapping before or during computation")
    else:
        print("Unequivalence for weight mapping before or during")
    
    print("For longest path in big matrix:", execute_simulator(big_matrix, root=0, final=9))
    print("For shortest path in big matrix:", execute_simulator(big_matrix, root=0, final=1))
    print("All duplicates of 5 yields", execute_simulator(all_duplicate, root=0, final=4))


edge_cases()

# in case of incorrect inputs, the correct error message must be displayed:
# if the input matrix is not a tree (it has loops), it should return error message
loop = np.array([[ 0,  0,  9,  2,  0],
                        [ 0,  0,  0,  5,  0],
                        [ 0,  0,  0,  0,  5],
                        [ 0,  0,  0,  0,  8],
                        [ 0,  0,  0,  0,  0]])
    
unconnected = np.array([[ 0,  0,  9,  2,  0],
                        [ 0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  5],
                        [ 0,  0,  0,  0,  8],
                        [ 0,  0,  0,  0,  0]])

negative =      np.array([[ 0,  -9,  10,  0,  7],
                              [ 0,  0,  0,  0,  0],
                              [ 0,  0,  0,  2,  0],
                              [ 0,  0,  0,  0,  0],
                              [ 0,  0,  0,  0,  0]])


# to test, these must be executed one by one, as each will yield an error message leading to sys.exit()

#execute_simulator(loop, root=0, final=1) # should give error: cycles
#execute_simulator(unconnected, root=0, final=1) # should give error: unconnected node
#execute_simulator(negative, root=0, final=1) # should give error: negative weight

#cap, path = execute_simulator(big_matrix, root=0, final=8)
#print(f"Capacity:\n{cap}")
#print(f"Path:\n{path}")
