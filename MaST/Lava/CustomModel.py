# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.lif.CustomProcess import LIF


class AbstractPyLifModelFloat(PyLoihiProcessModel):
    """Abstract implementation of floating point precision
    leaky-integrate-and-fire neuron model.

    Specific implementations inherit from here.
    """

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out = None  # This will be an OutPort of different LavaPyTypes
    v: np.ndarray = LavaPyType(np.ndarray, float)
    dv: float = LavaPyType(float, float)
    initial_v: np.ndarray = LavaPyType(np.ndarray, float)

    def spiking_activation(self):
        """Abstract method to define the activation function that determines
        how spikes are generated.
        """
        raise NotImplementedError(
            "spiking activation() cannot be called from "
            "an abstract ProcessModel"
        )

    def subthr_dynamics(self, activation_in: np.ndarray):
        """Common sub-threshold dynamics of current and voltage variables for
        all LIF models. This is where the 'leaky integration' happens.
        """
        self.v[:] = np.maximum(0, self.v * (1 - self.dv) + activation_in)
        
    def reset_voltage(self, spike_vector: np.ndarray):
        """Voltage reset behaviour. This can differ for different neuron
        models."""
        self.v[spike_vector] = self.initial_v

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        super().run_spk()
        a_in_data = self.a_in.recv()

        self.subthr_dynamics(activation_in=a_in_data)
        self.s_out_buff = self.spiking_activation()
        self.reset_voltage(spike_vector=self.s_out_buff)
        self.s_out.send(self.s_out_buff)


@implements(proc=LIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLifModelFloat(AbstractPyLifModelFloat):
    """Implementation of Leaky-Integrate-and-Fire neural process in floating
    point precision. This short and simple ProcessModel can be used for quick
    algorithmic prototyping, without engaging with the nuances of a fixed
    point implementation.
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def spiking_activation(self):
        """Spiking activation function for LIF."""
        return self.v >= self.vth




