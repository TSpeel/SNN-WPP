import numpy as np
from CustomNetwork import Network

class Constructor():
    def __init__(self, weight_matrix: np.ndarray) -> None:
        self.weight_matrix = weight_matrix
        self.nr_nodes, _ = weight_matrix.shape
        self.nr_edges = np.count_nonzero(weight_matrix)
        self.increase = np.power(2, self.nr_edges)
        self.target_nodes = []  
        self.graph_nodes = []
        self.max_weight = 0
        self.nr_steps = 0
        
    def construct(self):
        self.network = Network()
        
        # Create start node for starting each cycle.
        start_node = self.network.createLIF(dv=1, v=0, vth=1, read_out=True, ID=1)
        self.target_nodes.append(start_node)

        # Create add node for adding new nodes to the MST and suppressing the rest of the graph.
        add_node = self.network.createLIF(dv = 1, v = 0, vth =1, read_out=False, ID=2)
        self.network.createSynapse(add_node, add_node, w=-1, d=0) # Inhibit further spikes to the add node for 3 time-steps while the add node inhibits all the graph nodes
        self.network.createSynapse(add_node, add_node, w=-1, d=1)
        self.network.createSynapse(add_node, add_node, w=-1, d=2)

        id_counter = 3
        
        for i in range(self.nr_nodes):

            # Create graph node.
            graph_node = self.network.createLIF(dv = 0, v = 0, vth = self.nr_nodes, read_out=False, ID=id_counter)
            self.graph_nodes.append(graph_node)

            # At the start of a cycle, reset the graph nodes.
            self.network.createSynapse(start_node, graph_node, w=-self.nr_nodes, d=0)
            
            self.network.createSynapse(start_node, graph_node, w=self.nr_nodes-1, d=1)

            # If a node is added to the MST, inhibit all other graph nodes from spiking.
            self.network.createSynapse(add_node, graph_node, w=-self.nr_nodes, d=0)

            # Create a node for keeping track if the node is already part of the MST.
            part_of_mst_node = self.network.createLIF(dv = 0, v = (0 if i else self.nr_nodes), vth = self.nr_nodes + 1, read_out=False, ID=id_counter+1)
            self.network.createSynapse(part_of_mst_node, part_of_mst_node, w = self.nr_nodes, d = 0)
                 
            # Create an AND-gate to check if a node is already part of the MST when a new cycle starts. If so, spike the node.
            and_gate_part_of_mst_node = self.network.createLIF(dv = 1, v = 0, vth=2, read_out=False, ID=id_counter+2)
            self.network.createSynapse(start_node, part_of_mst_node, w=1, d=0)
            self.network.createSynapse(part_of_mst_node, and_gate_part_of_mst_node, w=1, d=0)
            self.network.createSynapse(start_node, and_gate_part_of_mst_node, w=1, d=1)
            self.network.createSynapse(and_gate_part_of_mst_node, graph_node, w=1, d=0)

            # Create a node to check if the graph node is not yet part of the MST, and thus should be added. If so, spike the add node.
            add_to_mst_node = self.network.createLIF(dv = 1, v = 0, vth = 1, read_out=False, ID=id_counter+4)
            self.network.createSynapse(graph_node, part_of_mst_node, w=1, d=0)
            self.network.createSynapse(part_of_mst_node, add_to_mst_node, w=-1, d=0)
            self.network.createSynapse(graph_node, add_to_mst_node, w=1, d=1)
            self.network.createSynapse(add_to_mst_node, add_node, w=1, d=0)
            self.network.createSynapse(add_to_mst_node, part_of_mst_node, w=-1, d=0)

            # Create an AND-gate to check if the graph node caused the add node to spike. If so, spike part of MST node to add it to the MST.
            and_gate_add_to_mst_node = self.network.createLIF(dv = 1, v = 0, vth = 2, read_out=True, ID=id_counter+3)
            self.target_nodes.append(and_gate_add_to_mst_node)
            self.network.createSynapse(add_node, and_gate_add_to_mst_node, w=1, d=0)
            self.network.createSynapse(graph_node, and_gate_add_to_mst_node, w=1, d=3)
            self.network.createSynapse(and_gate_add_to_mst_node, part_of_mst_node, w=self.nr_nodes + 1, d=0)

            id_counter += 5

        
        counter = 1
        self.weight_matrix_transformed = np.zeros((self.nr_nodes, self.nr_nodes))
        unique_weights, weight_counts = np.unique(self.weight_matrix, return_counts=True)
        max_duplicate_weights = weight_counts[1] if 0 in unique_weights else weight_counts[0]
        
        # For every connection, add a synapse with delay representing weight
        weight_history = []
        for i in range(self.nr_nodes):
            for j in range(i + 1, self.nr_nodes):
                w = self.weight_matrix[i,j]
                if i != j and w > 0:
                    weight_count = np.count_nonzero(weight_history == w)
                    weight = int(max_duplicate_weights * w + weight_count)
                    weight_history.append(w)
                    self.network.createSynapse(self.graph_nodes[i], self.graph_nodes[j], w = 1, d = weight-1)
                    self.network.createSynapse(self.graph_nodes[j], self.graph_nodes[i], w = 1, d = weight-1)
                    self.weight_matrix_transformed[i,j] = weight

                    # Increase counter
                    counter+=1

                    # Keep track of maximum delay
                    if weight > self.max_weight:
                        self.max_weight = weight


        # Create reset node for resetting the algorithm and starting the next cycle.
        # Add delay of maximum weight to make sure all pulses have propagated
        self.network.createSynapse(add_node, start_node, w = 1, d = self.max_weight + 2)
        
        self.num_steps = (self.nr_nodes-1) * (self.max_weight*2 + 8) + 1
        
        # Create train to start the algorithm.
        self.network.createStartTrain(start_node, num_steps = self.num_steps)
        
        return self.network

    
def execute_simulator(weight_matrix):
    constructor = Constructor(weight_matrix)
    network = constructor.construct()

    weights = constructor.weight_matrix_transformed

    network.monitor_processes(num_steps=constructor.num_steps)
    
    network.run(num_steps=constructor.num_steps)
    return network.construct_MST(weights, weight_matrix)
    
    
weight_matrix = np.array([
    [0, 1, 5, 0, 9], # A
    [0, 0, 2, 7, 0], # B
    [0, 0, 0, 3, 1], # C
    [0, 0, 0, 0, 10], # D
    [0, 0, 0, 0, 0] # E
])

weight_matrix_ones = np.array([
    [0, 1, 1, 0, 1], # A
    [0, 0, 1, 1, 1], # B
    [0, 0, 0, 1, 1], # C
    [0, 0, 0, 0, 1], # D
    [0, 0, 0, 0, 0] # E
])

mst_matrix = execute_simulator(weight_matrix)

print(f"Graph: {weight_matrix}")
print(f"MST: {mst_matrix}")

    
            
            
            
