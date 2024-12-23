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
    def __init__(self, weight_matrix: np.ndarray) -> None:
        self.weight_matrix = weight_matrix
        self.nr_nodes, _ = weight_matrix.shape
        self.nr_edges = np.count_nonzero(weight_matrix)
        self.target_nodes = []
        self.graph_nodes = []
        self.receive_graph_nodes = []
        self.max_weight = 0

    def construct(self,max_delay):
        self.network = Network()

        # Create start node for starting each cycle.
        start_node = self.network.createLIF(m=0, V_init=0, V_reset=0, thr=1, read_out=True, ID=1)
        self.target_nodes.append(start_node)

        # Create train to start the algorithm.
        start_train = self.network.createInputTrain([1], loop=False, ID=2)
        self.network.createSynapse(start_train, start_node, w=1, d=1)


        # Create add node for adding new nodes to the MST. 
        # Inhibition is no longer needed as last spike in the network is the one triggering the addition of the edge.
        add_node = self.network.createLIF(m=1, V_init=self.nr_edges, V_reset=self.nr_edges, thr=self.nr_edges+1, read_out=True, ID=3)
        self.target_nodes.append(add_node)
        self.network.createSynapse(add_node, add_node, w=-self.nr_edges, d=1)
        self.network.createSynapse(add_node, add_node, w=self.nr_edges, d=2)

        #Node responsible for adding +1 to count of relevant edges in cycle. Has to ping after first -1 ping once per cycle.
        adjust_count_node = self.network.createLIF(m=1, V_init=self.nr_edges, V_reset=0, thr=self.nr_edges+1, read_out=False, ID=4)
        self.network.createSynapse(adjust_count_node, add_node, w=1, d=1)
        

        id_counter = 5

        for i in range(self.nr_nodes):

            # Create graph node.
            graph_node = self.network.createLIF(m=1, V_init=0, V_reset=0, V_min=0, thr=self.nr_nodes, read_out=False, ID=id_counter)
            self.graph_nodes.append(graph_node)

            receive_graph_node = self.network.createLIF(m=1, V_init=0, V_reset=0, V_min=0, thr=1, read_out=False, ID=id_counter)
            self.receive_graph_nodes.append(receive_graph_node)

            # At the start of a cycle, reset the graph nodes.
            self.network.createSynapse(start_node, graph_node, w=-self.nr_nodes, d=2)
            self.network.createSynapse(start_node, graph_node, w=self.nr_nodes-1, d=3)

            # If a node is added to the MST, inhibit all other graph nodes from spiking.
            self.network.createSynapse(add_node, graph_node, w=-self.nr_nodes, d=1)
            self.network.createSynapse(add_node, receive_graph_node, w=-self.nr_nodes, d=1)

            # Create a node for keeping track if the node is already part of the MST.
            part_of_mst_node = self.network.createLIF(m=1, V_init=(0 if i else self.nr_nodes), V_reset=0, thr=self.nr_nodes + 1, read_out=True, ID=id_counter+1)
            self.network.createSynapse(part_of_mst_node, part_of_mst_node, w=self.nr_nodes, d=1)
            
            # Create an AND-gate to check if a node is already part of the MST when a new cycle starts. If so, spike the node.
            and_gate_part_of_mst_node = self.network.createLIF(m=0, V_init=0, V_reset=0, thr=2, read_out=False, ID=id_counter+2)
            self.network.createSynapse(start_node, part_of_mst_node, w=1, d=2)
            self.network.createSynapse(part_of_mst_node, and_gate_part_of_mst_node, w=1, d=1)
            self.network.createSynapse(start_node, and_gate_part_of_mst_node, w=1, d=3)
            self.network.createSynapse(and_gate_part_of_mst_node, graph_node, w=1, d=1)

            # Create a node to check if the graph node is not yet part of the MST, and thus should be added. If so, spike the add node.
            #v_notpart
            v_notpart_node = self.network.createLIF(m=0, V_init=0, V_reset=0, thr=1, read_out=False, ID=id_counter+4)
            self.network.createSynapse(receive_graph_node, part_of_mst_node, w=1, d=1)
            self.network.createSynapse(part_of_mst_node, v_notpart_node, w=-1, d=1)
            self.network.createSynapse(receive_graph_node, v_notpart_node, w=1, d=2)
            self.network.createSynapse(v_notpart_node, add_node, w=1, d=max_delay+1)
            self.network.createSynapse(v_notpart_node, part_of_mst_node, w=-1, d=1)

            # Counting number of edges from MST to non-MST nodes
            self.network.createSynapse(v_notpart_node, add_node, w=-1, d=1)
            self.network.createSynapse(v_notpart_node, adjust_count_node, w=1, d=1)


            # Create an AND-gate to check if the graph node caused the add node to spike. If so, spike part of MST node to add it to the MST.
            and_gate_v_notpart_node = self.network.createLIF(m=0, V_init=0, V_reset=0, thr=2, read_out=False, ID=id_counter+3)
            self.target_nodes.append(and_gate_v_notpart_node)
            self.network.createSynapse(add_node, and_gate_v_notpart_node, w=1, d=1)
            self.network.createSynapse(receive_graph_node, and_gate_v_notpart_node, w=1, d=max_delay+4)
            self.network.createSynapse(and_gate_v_notpart_node, part_of_mst_node, w=self.nr_nodes + 1, d=1)

            id_counter += 5

        
        counter = 1
        self.weight_matrix_transformed = np.zeros((self.nr_nodes, self.nr_nodes))
        unique_weights, weight_counts = np.unique(self.weight_matrix, return_counts=True)
        max_duplicate_weights = weight_counts[1] if 0 in unique_weights else weight_counts[0]

        # For every connection, add a synapse with delay representing weight
        weight_history = []
        for i in range(self.nr_nodes):
            for j in range(i+1, self.nr_nodes):
                w = self.weight_matrix[i,j]
                if i != j and w > 0:
                    weight_count = np.count_nonzero(weight_history == w)
                    weight = int(max_duplicate_weights * w + weight_count)
                    weight_history.append(w)
                    # Synapse goes to receive-only neuron to prevent further spikes from non-MST to non-MST nodes
                    self.network.createSynapse(self.graph_nodes[i], self.receive_graph_nodes[j], w=1, d=weight)
                    self.network.createSynapse(self.graph_nodes[j], self.receive_graph_nodes[i], w=1, d=weight)
                    self.weight_matrix_transformed[i,j] = weight


                    # Increase counter
                    counter+=1

                    # Keep track of maximum delay
                    if weight > self.max_weight:
                        self.max_weight = weight

        # Create reset node for resetting the algorithm and starting the next cycle.
        self.network.createSynapse(add_node, start_node, w=1, d=3)
        self.network.createSynapse(add_node, adjust_count_node, w=-self.nr_edges, d=1)
        self.network.createSynapse(add_node, adjust_count_node, w=self.nr_edges, d=2)

        return self.network


def construct_MST(spikes, delays, weight_matrix,max_delay):
    cycle_starts = np.argwhere(spikes[:, 0] == True).flatten()
    #print(cycle_starts)
    #print([np.argwhere(spikes[:, node]==True).flatten() for node in range(3, spikes.shape[1])])
    #spikes_at = [np.argwhere(spikes[:, node]==True).flatten()[0] for node in range(3, spikes.shape[1])]
    spikes_at = [np.flatnonzero(spikes[:, node])[0] if spikes[:, node].any() else 99999 for node in range(3, spikes.shape[1])]

    mst_matrix = np.zeros(weight_matrix.shape)
    #print(delays)
    for i, spike_time in enumerate(sorted(spikes_at)):
        start_cycle = cycle_starts[i]
        
        time_needed = spike_time - start_cycle - 8 - max_delay
        #print(spike_time,start_cycle,time_needed)
        result = np.argwhere(delays==time_needed).flatten()
        mst_matrix[result[0], result[1]] = weight_matrix[result[0], result[1]]

    return mst_matrix

def execute_simulator(weight_matrix,max_delay):
    constructor = Constructor(weight_matrix)
    network = constructor.construct(max_delay)

    weights = constructor.weight_matrix_transformed

    sim = Simulator(network)
    
    # Add all read_out neurons to the simulation
    sim.raster.addTarget(constructor.target_nodes)
    sim.multimeter.addTarget(constructor.target_nodes)

    sim.run(steps=(constructor.nr_nodes-1) * (constructor.max_weight*2 + 8) + 1 + 2*constructor.nr_nodes*max_delay, plotting=False)
    # Obtain all measurements
    spikes = sim.raster.get_measurements()
    voltages = sim.multimeter.get_measurements()
    #print(voltages[:,1])

    return construct_MST(spikes, weights, weight_matrix,max_delay)


parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str)
args = parser.parse_args()
if args.i is not None:
    weight_matrix = np.array(ast.literal_eval(args.i))
else:
    weight_matrix = np.array([
        [0, 1, 5, 0, 9], # A
        [0, 0, 2, 7, 0], # B
        [0, 0, 0, 3, 1], # C
        [0, 0, 0, 0, 14], # D
        [0, 0, 0, 0, 0] # E
    ])
#Usage eg python3 createMaST.py -i [[0,1,5,0,9],[0,0,2,7,0],[0,0,0,3,1],[0,0,0,0,14],[0,0,0,0,0]]
# or just python3 createMaST.py
print(f"Graph:\n{weight_matrix}\n")
weight_matrix *= 2
max_delay = np.max(weight_matrix) * 2
mst_matrix = execute_simulator(weight_matrix,max_delay)/2

print(f"MaST:\n{mst_matrix}")