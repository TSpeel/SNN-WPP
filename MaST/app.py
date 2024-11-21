from flask import Flask, request, jsonify
from flask_cors import cross_origin

import warnings
import traceback
from numpy import inf

from core.simulators import Simulator
from core.converters import Deserializer

# to make sure that warnings of server are not ignored in response to user
warnings.filterwarnings("error")
warnings.filterwarnings("ignore", message="Partition None overridden by environment variable PARTITION=oheogulch")


def process_measurements(spikes, voltages, network, json):
    measurements = {  }
    for i, node in enumerate(network.nodes):
        measurements[node.ID] = { 
            "name": json['nodes'][i].get("name"), 
            "spikes": [], 
            "voltages": [], 
            "read_out": json['nodes'][i].get("read_out")
        }

        for spike in spikes:
            measurements[node.ID]["spikes"].append(str(spike[i]).lower())
        
        for voltage in voltages:
            if voltage[i] == inf:
                measurements[node.ID]["voltages"].append("\u221E")
            else:
                measurements[node.ID]["voltages"].append(voltage[i])

    return measurements


def execute_simulator(network, duration, json):
    sim = Simulator(network)
    
    # Add all read_out neurons to the simulation
    sim.raster.addTarget(network.nodes)
    sim.multimeter.addTarget(network.nodes)

    sim.run(steps=duration, plotting=False)

    # Obtain all measurements
    spikes = sim.raster.get_measurements()
    voltages = sim.multimeter.get_measurements()

    measurements = process_measurements(spikes, voltages, network, json)
    
    return measurements


def execute_loihi(network, duration, json):
    from compiler.executor import Executor

    executor = Executor(network, steps=duration)
    executor.run()

    voltages = executor.voltages
    spikes = executor.spikes

    measurements = {}
    for node in json['nodes']:
        node_id = node.get("id")
        measurements[node_id] = { 
            "name": node.get("name"), 
            "spikes": [str(spike).lower() for spike in spikes[node_id]], 
            "voltages": voltages[node_id], 
            "read_out": node.get("read_out")
        }

    return measurements


def compare_voltages(simulator_voltages, loihi_voltages, name):
    assert len(simulator_voltages) == len(loihi_voltages)
    
    for i in range(len(simulator_voltages)):
        if abs(simulator_voltages[i] - loihi_voltages[i]) > 0.1:
            print("Voltage does not match for Loihi V =",loihi_voltages[i]," and the simulator V =",simulator_voltages[i],"for Node", name, " at time step", i)
            return False
        
    return True


def compare_spikes(simulator_spikes, loihi_spikes, name):
    # if loihi did not record spikes: return
    if len(loihi_spikes) == 0:
        return True
    
    assert len(simulator_spikes) == len(loihi_spikes)
    
    for i in range(len(simulator_spikes)):
        if simulator_spikes[i] != loihi_spikes[i]:
            print("Spike behaviour does not match for Loihi and the simulator for Node", name, " at time step", i)
            return False
        
    return True
     

def match_results(simulator_measurements, loihi_measurements):
    for node_id, node in loihi_measurements.items():
        simulator_voltages = simulator_measurements[node_id]["voltages"]
        if not compare_voltages(simulator_voltages, node["voltages"], node["name"]):
            return False
        
        simulator_spikes = simulator_measurements[node_id]["spikes"]
        if not compare_spikes(simulator_spikes, node["spikes"], node["name"]):
            return False
        
    return True


app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/', methods=['POST,GET'])
@cross_origin()
def network():
    json = request.get_json()

    try:
        network = Deserializer(json).network
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Build error: " + str(e) + ". Look at server output for traceback."})

    try:
        duration = int(json["duration"])
    except ValueError:
        duration = 10

    loihi = json["config"]["loihi"]["loihiExecution"]
    
    try:
        if loihi:
            print("INFO: Started execution on Loihi")
            measurements = execute_loihi(network, duration, json)
            print("INFO: Finished execution on Loihi")
            
            if json["config"]["loihi"]["matchWithSimulator"]:

                # give warning if network contains random generator without seed (and this generator is used)
                if any(node["type"] == "random" and node["rng"] == "" for node in json["nodes"]):
                    print("Warning: trying to match results of Loihi with simulator having random spiker without specified seed.")

                # check whether Loihi results match that of the simulator
                new_network = Deserializer(json).network
                simulator_measurements = execute_simulator(new_network, duration, json)
                if not match_results(simulator_measurements, measurements):
                    return jsonify({"error": "Execution error: measurements of Loihi do not match with the simulator. Look at server output for traceback."})
                else:
                    print("Measurements for Loihi match with the simulator")
        else:
            print("INFO: Started execution on simulator")
            measurements = execute_simulator(network, duration, json)
            print("INFO: Finished execution on simulator")
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Execution error: " + str(e) + ". Look at server output for traceback."})

    return jsonify({"duration": duration, "loihi": str(loihi).lower(), "measurements": measurements, "error": None})