import openpnm as op
import numpy as np
import topography 
import simulation
import transport
from parameters import input_dict as params

ws = op.utils.Workspace()
project = ws.load_project(filename='17x_vertical_Freudenberg.pnm')
network = project.network

network = topography.segment_electrode(network=network)
network = topography.create_ff_pores(network=network)
network = topography.add_ff_to_electrode_throats(network=network)
network = topography.add_ff_to_ff_throats(network=network)
network = topography.define_membrane(network=network)
network = topography.define_ff_boundaries(network=network)

ws.save_project(project, filename='segmented_freudenberg.pnm')
op.io.project_to_vtk(project, filename="yerrrrrr")

# ==================== CREATE PHASES ====================
anolyte, catholyte = simulation.create_phases(network, params)

# ==================== CALCULATE INLET FLOW RATE ====================
v_in_c = params['catholyte_inlet_velocity']
inlet_pores = network.pores('inlet')
cross_area = np.pi / 4 * network['pore.diameter']**2
A_in_c = np.sum(cross_area[inlet_pores])
Q_in_c = v_in_c * A_in_c

# ==================== SETUP STOKES FLOW ====================
sf_catholyte = simulation.setup_stokes_flow(network, catholyte, Q_inlet=Q_in_c, P_outlet=0.0)

# ==================== EXPORT ====================
project.update({'catholyte': catholyte})
ws.save_project(project, filename='stokes_test.pnm')
op.io.project_to_vtk(project, filename="stokes_test")