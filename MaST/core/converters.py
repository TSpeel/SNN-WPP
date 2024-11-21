from numpy import inf
from numpy.random import RandomState

from core.networks import Network

    
class Deserializer:
  def __init__(self, json):
      self.network = self.deserialize_network(json)


  def deserialize_network(self, json):
      network = Network()
      nodes = {}
      for json_node in json['nodes']:
          node = self.create_node(network, json_node)
          nodes[node.ID] = node

      for json_synapse in json['synapses']:
          self.create_synapse(network, nodes, json_synapse)

      return network
  

  def create_node(self, network, node):
      id = node['id']
      if node['type'] == "lif":
          params = {
              "ID": id,
              "m": self.parse_float(node['m']),
              "V_init": self.parse_float(node['V_init']),
              "V_reset": self.parse_float(node['V_reset']),
              "V_min": self.parse_float(node['V_min']),
              "thr": self.parse_float(node['thr']),
              "amplitude": self.parse_float(node['amplitude']),
              "I_e": self.parse_float(node['I_e']),
              "noise": self.parse_float(node['noise'])
          }
          
          if node['rng']:
              params["rng"] = RandomState(self.parse_int(node['rng']))
          
          return network.createLIF(**params)
      elif node['type'] == "input":
          params = {
              "ID": id,
              "train": self.parse_float_list(node['train']),
              "loop": bool(node['loop'])
          }

          return network.createInputTrain(**params)
      elif node['type'] == "random":
          params = {
              "ID": id,
              "p": self.parse_float(node['p']),
              "amplitude": self.parse_float(node['amplitude'])
          }

          if node['rng']:
              params["rng"] = RandomState(self.parse_int(node['rng']))

          return network.createRandomSpiker(**params)  


  def create_synapse(self, network, nodes, synapse):
      pre = nodes[synapse['pre']]
      post = nodes[synapse['post']]
      w = self.parse_float(synapse['w'])
      d = self.parse_int(synapse['d'])

      return network.createSynapse(pre=pre, post=post, w=w, d=d)


  def parse_float(self, value):
      if value == "\u221E":
          return inf
      
      if value == "-\u221E":
          return -inf
      
      return float(value)


  def parse_float_list(self, value):
      # pre-process: remove brackets
      value = value.replace("[", "")
      value = value.replace("]", "")

      # split on ,
      floats = value.split(",")

      return list(map(lambda x: self.parse_float(x), floats))


  def parse_int(self, value):
      if value == "\u221E":
          return inf
      
      if value == "-\u221E":
          return -inf
      
      return int(value)


  
