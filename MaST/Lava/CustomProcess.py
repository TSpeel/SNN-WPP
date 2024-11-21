# Copyright (C) 2022-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.magma.core.process.process import LogConfig, AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class AbstractLIF(AbstractProcess):
    """Abstract class for variables common to all neurons with leaky
    integrator dynamics."""

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        v: ty.Union[float, list, np.ndarray],
        dv: float,
        initial_v: ty.Union[float, list, np.ndarray],
        name: str,
        log_config: LogConfig,
        **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            initial_v=initial_v,
            v=v,
            dv=dv,
            name=name,
            log_config=log_config,
            **kwargs,
        )

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.v = Var(shape=shape, init=v)
        self.dv = Var(shape=(1,), init=dv)
        self.initial_v = Var(shape=(1,), init=initial_v)

class LIF(AbstractLIF):
    """Leaky-Integrate-and-Fire (LIF) neural Process.

    LIF dynamics abstracts to:
    u[t] = u[t-1] * (1-du) + a_in         # neuron current
    v[t] = v[t-1] * (1-dv) + u[t] + bias  # neuron voltage
    s_out = v[t] > vth                    # spike if threshold is exceeded
    v[t] = 0                              # reset at spike

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    u : float, list, numpy.ndarray, optional
        Initial value of the neurons' current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    du : float, optional
        Inverse of decay time-constant for current decay. Currently, only a
        single decay can be set for the entire population of neurons.
    dv : float, optional
        Inverse of decay time-constant for voltage decay. Currently, only a
        single decay can be set for the entire population of neurons.
    bias_mant : float, list, numpy.ndarray, optional
        Mantissa part of neuron bias.
    bias_exp : float, list, numpy.ndarray, optional
        Exponent part of neuron bias, if needed. Mostly for fixed point
        implementations. Ignored for floating point implementations.
    vth : float, optional
        Neuron threshold voltage, exceeding which, the neuron will spike.
        Currently, only a single threshold can be set for the entire
        population of neurons.

    Example
    -------
    >>> lif = LIF(shape=(200, 15), du=10, dv=5)
    This will create 200x15 LIF neurons that all have the same current decay
    of 10 and voltage decay of 5.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        initial_v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        dv: ty.Optional[float] = 0,
        vth: ty.Optional[float] = 10,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            initial_v = initial_v,
            v=v,
            dv=dv,
            name=name,
            log_config=log_config,
            **kwargs,
        )

        self.vth = Var(shape=(1,), init=vth)


