#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/11 14:29
# @Author  : Ws
# @File    : forward.py
# @Software: PyCharm

from simpeg import maps
import numpy as np
from simpeg.electromagnetics import time_domain as tdem

from retrieval.library import AEMLibraryWriter

layer_thicknesses = np.asarray(
    [5, 5.4, 5.8, 6.3, 6.7, 7.3, 7.8, 8.4, 9.1, 9.8, 10.6, 11.4, 12.3, 13.2, 14.3, 15.4, 16.6, 17.9, 19.3, 20.7, 22.4,
     24.1, 26, 28, 30.2, 32.5, 35, 37.8, 40.7])


def forward(rho, layer_thicknesses, h, times):
    hcustom_times = np.array([
        -4.00000e-003,
        -3.91300e-003,
        -3.81313e-003,
        -3.72277e-003,
        -3.68473e-003,
        -2.30082e-003,
        -1.10204e-003,
        0.00000e+000,
        3.25433e-006,
        9.99755e-005,
        2.01631e-004,
        2.81574e-004,
        3.08222e-004,
        3.13115e-004,
        3.14989e-004,
        3.16781e-004,
        3.18848e-004
    ])

    hcustom_currents = np.array([
        0.00000e+000,
        3.17153e-001,
        6.30292e-001,
        8.79197e-001,
        9.61095e-001,
        9.73942e-001,
        9.88394e-001,
        1.00000e+000,
        9.91368e-001,
        7.01860e-001,
        3.77911e-001,
        1.15687e-001,
        2.79070e-002,
        1.21241e-002,
        6.60584e-003,
        3.02920e-003,
        0.00000e+000
    ])
    hm_wave = tdem.sources.PiecewiseLinearWaveform(hcustom_times, hcustom_currents)
    tx_shape = np.array([[-12.64,-2.13, 0.], [-6.15, -8.59, 0.], [5.74, -8.59, 0.],
                         [11.13, -3.19, 0.], [ 11.13, 3.19, 0.], [5.74, 8.59, 0.],
                         [-6.15, 8.59, 0.], [-12.64, 2.13, 0.], [-12.64, -2.13, 0.]])
    tx_shape[-1] = tx_shape[0]
    rx_offset = np.array([-13.25, 0, -2.0])
    tx_loc = tx_shape + [0, 0, h]
    rx_loc = rx_offset + [0, 0, h]

    rx_hm = tdem.receivers.PointMagneticFluxTimeDerivative(
        rx_loc, times, orientation='z'
    )

    src_hm = tdem.sources.LineCurrent([rx_hm], tx_loc, waveform=hm_wave)

    survey = tdem.Survey([src_hm])

    n_layers = len(rho)
    model = 1 / rho
    model_mapping = maps.IdentityMap(nP=n_layers)

    # Define the simulation
    simulation = tdem.Simulation1DLayered(
        survey=survey,
        thicknesses=layer_thicknesses,
        sigmaMap=model_mapping,
    )

    dpred = simulation.dpred(model)
    return np.abs(dpred) / 341


def forward_and_store(
    writer,
    rho,
    layer_thicknesses,
    h,
    times,
    sample_id=None,
):
    """Run one forward simulation and immediately route it into the library."""

    response = forward(rho, layer_thicknesses, h, times)
    index = writer.add(response, height_m=h, model=rho, sample_id=sample_id)
    return response, index

def generate_library(
    models,
    heights,
    times,
    output,
    thicknesses=layer_thicknesses,
    layout_config=None,
    progress_every=1000,
    forward_operator=None,
):
    """Generate responses and their final retrieval organization in one pass.

    ``forward_operator`` is injectable for testing; production defaults to the
    SimPEG-backed :func:`forward` above.
    """

    model_matrix = np.asarray(models)
    height_values = np.asarray(heights)
    time_values = np.asarray(times, dtype=float).reshape(-1)

    writer = AEMLibraryWriter(
            output,
            capacity=model_matrix.shape[0],
            n_model_parameters=model_matrix.shape[1],
            config=layout_config,
        )

    operator = forward if forward_operator is None else forward_operator
    for index in range(writer.count, model_matrix.shape[0]):
        height = height_values[index]
        response = operator(
            model_matrix[index], thicknesses, height, time_values
        )
        writer.add(
            response,
            height_m=height,
            model=model_matrix[index],
            sample_id=index,
        )
        if progress_every and writer.count % int(progress_every) == 0:
            print(f"generated and routed {writer.count}/{writer.capacity}")
    writer.finalize()
    path = writer.path
    writer.release()
    return path

