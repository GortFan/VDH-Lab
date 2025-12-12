import openpnm as op
import topography 

ws = op.utils.Workspace()
project = ws.load_project(filename='17x_vertical_Freudenberg.pnm')
network = project.network

network = topography.segment_electrode(network=network)
network = topography.create_ff_pores(network=network)
network = topography.connect_ff_pores_eachother(network=network)
network = topography.connect_electrode_to_ff_pores(network=network)
network = topography.define_membrane(network=network)
network = topography.define_ff_boundaries(network=network)
network = topography.define_ff_internal_channel_pores(network=network)
network = topography.create_current_collector_pores(network=network)
ws.save_project(project, filename='ff_freudenberg')
op.io.project_to_vtk(project, filename="ff_freudenberg")