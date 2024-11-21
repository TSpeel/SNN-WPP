
from lava.proc.lif.CustomProcess import LIF
from lava.proc.dense.process import DelayDense
import numpy as np
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.monitor.process import Monitor
from lava.proc.io.source import RingBuffer

class Network:

    def __init__(self, nodes=None, synapses=None):
        self.nodes = nodes if nodes is not None else []
        self.synapses = synapses if synapses is not None else []
        self.monitor_list = []
        self.spikes = []
        self.read_out_nodes = []
        self.pattern_pre = None
        self.v_list = []
        self.voltages = []
    
    # define spike train 
    # set first spike to 1 to start the algorithm  
    def createStartTrain(self, start_node, num_steps=0):
        spike_raster = np.zeros((1,num_steps))
        spike_raster[0,0] = 1
        self.pattern_pre = RingBuffer(data = spike_raster) 
        # Create input connectivity
        self.createSynapse(self.pattern_pre, start_node, d = 0, w = 1)
        
    #create LAVA nodes and store the target nodes  
    def createLIF(self, dv = 1, v = 0, vth = 1, ID = "", read_out = False):
        node = LIF(shape = (1,), v = v, vth = vth, dv = dv, name = str(ID))
        self.nodes.append(node)
        if(read_out):
            self.read_out_nodes.append(node)
        
        return node
    
    #create LAVA synapse connecting pre- and postneuron
    def createSynapse(self, pre_neuron, post_neuron, w = 1, d = 1):
        weight = np.eye(1) * w
        synapse = DelayDense(weights = weight, delays = d)
        pre_neuron.s_out.connect(synapse.s_in)
        synapse.a_out.connect(post_neuron.a_in)
        self.synapses.append(synapse)
        
        return synapse  

    #start the algorithm by starting the input train 
    #monitor the target neuron spikes
    def run(self, num_steps):
        
        self.pattern_pre.run(condition=RunSteps(num_steps=num_steps), run_cfg=Loihi1SimCfg())
        for i in range(len(self.monitor_list)):
            self.spikes.append(self.monitor_list[i].get_data()[self.read_out_nodes[i].name]['s_out'])
        
        self.pattern_pre.stop()
        
    #extract the MST from the spikes from target neurons 
    def construct_MST(self, delays, weight_matrix):
        spikes = np.array(self.spikes)
        spikes = spikes[:,:,0].T
        cycle_starts = np.argwhere(spikes[:, 0] == True).flatten()
            
        spikes_at = [np.argwhere(spikes[:, node]==True).flatten()[0] for node in range(2, spikes.shape[1])]
        mst_matrix = np.zeros(weight_matrix.shape)
        
        for i, spike_time in enumerate(sorted(spikes_at)):
            start_cycle = cycle_starts[i]
            time_needed = spike_time - start_cycle - 7
            result = np.argwhere(delays == time_needed).flatten()
            mst_matrix[result[0], result[1]] = weight_matrix[result[0], result[1]]
        return mst_matrix
    
    # create List to record the spikes of target neurons  
    def monitor_processes(self, num_steps):
        for each in self.read_out_nodes:
            trace = Monitor()
            trace.probe(each.s_out,num_steps)
            self.monitor_list.append(trace)
            
        return self.monitor_list
          
