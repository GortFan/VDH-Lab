import openpnm as op
import numpy as np
import query as q

if not hasattr(np, 'bool'):
    np.bool = bool
# step 1: segment left labeled pores into 2 symmetric groups about the y-axis midpoint

def segment_electrode(network: op.network.Network) -> op.network.Network:
    """
    Segments the electrode network boundary pores, labelled 'right' into two symmetric groups about the y-axis midpoint. 
    Segments will be labelled electrode_rib and electrode_ff, respectively. 

    Parameters:
    -----------
    pn : OpenPNM Network object
        The network that the operation will be applied to
    
    Returns:
    --------
    OpenPNM Network object : OpenPNM Network object with the segmentation changes applied.
    """
    pore_coords = network['pore.coords']
    right_pores_idx = network.pores('right')
    right_pores_coords = pore_coords[right_pores_idx]

    y_min = np.min(pore_coords, axis=0)[1]
    y_max = np.max(pore_coords, axis=0)[1]
    y_midpoint = (y_max - y_min) / 2

    left_pores_bisected = right_pores_coords[:,1] >= y_midpoint
    ff_idx = right_pores_idx[left_pores_bisected]
    rib_idx = right_pores_idx[~left_pores_bisected]

    network.set_label(label='rib', pores = rib_idx)
    network.set_label(label='channel', pores = ff_idx)

    return network

# step 2 (multistep): 
# 2.1: segment channel pores into 34 equally sized elements along the z-axis (do not label), just create generalized bounds
# 2.2: get center coordinates for each segment and collect into the following structure [[y,z], ...]
# 2.3: create 34 flow field pores - label = ff, diameter = y_size, location@center coordinate w/ x @ arbitrary dist
def create_ff_pores(network: op.network.Network) -> op.network.Network:
    """
    Creates 34 flow field, labelled 'ff' pores positioned along the z-axis at the channel region.
    Sets all required pore properties including diameter, volume, surface area, and centroid.
    
    Parameters
    ----------
    network : op.network.Network
        The network to add FF pores to
    
    Returns
    -------
    op.network.Network
        Network with FF pores added and labeled, with all properties defined.
    """
    pore_coords = network['pore.coords']
    channel_pores_idx = network.pores('channel')
    channel_pores_coords = pore_coords[channel_pores_idx]

    z_min, z_max = np.min(pore_coords, axis=0)[2], np.max(pore_coords, axis=0)[2]
    y_min, y_max = np.min(channel_pores_coords, axis=0)[1], np.max(channel_pores_coords, axis=0)[1]
    element_z_size, element_y_size = (z_max - z_min) / 34, (y_max - y_min)
    y_center = (y_min + y_max) / 2
    z_center = np.arange(start=z_min + element_z_size/2, stop=z_max, step=element_z_size)
    channel_pore_centroid_x = network['pore.centroid'][network.pores('channel')][0][0]
    channel_pore_diameter = network['pore.diameter'][network.pores('channel')][0]
    ff_pore_coordinates = [[channel_pore_centroid_x+(channel_pore_diameter/2)+(element_y_size/2), y_center, z] for z in z_center]
    
    op.topotools.extend(network, coords=ff_pore_coordinates)
    ff_pore_indices = np.arange(network.Np - len(ff_pore_coordinates), network.Np)
    network.set_label(label='ff', pores=ff_pore_indices)
    network['pore.diameter'][ff_pore_indices] = element_y_size
    network['pore.volume'][ff_pore_indices] = (4/3) * np.pi * (element_y_size/2)**3 
    network['pore.surface_area'][ff_pore_indices] = 4 * np.pi * (element_y_size/2)**2 
    network['pore.area'][ff_pore_indices] = np.pi * (element_y_size/2)**2
    network['pore.centroid'][ff_pore_indices] = ff_pore_coordinates
    return network

def connect_ff_pores_eachother(network: op.network.Network) -> op.network.Network:
    """
    Creates throat connections between sequential FF pores (FF1→FF2, FF2→FF3, etc.).
    Sets all required throat properties including diameter and length (tip-to-tip distance).
    Labels these throats as 'ff_to_ff_throat'.
    
    Parameters
    ----------
    network : op.network.Network
        The network with FF pores already created
    
    Returns
    -------
    op.network.Network
        Network with FF-to-FF throats added and labeled, with all properties defined.
    """
    ff_to_ff_conns = []
    ff_pore_indices = network.pores('ff')
    for i in range(len(ff_pore_indices) - 1):
        ff_to_ff_conns.append([ff_pore_indices[i], ff_pore_indices[i+1]])

    op.topotools.extend(network, conns=ff_to_ff_conns)
    ff_to_ff_throat_indices = np.arange(network.Nt - len(ff_to_ff_conns), network.Nt)
    network.set_label(label='ff_to_ff_throat', throats=ff_to_ff_throat_indices)
    network['throat.diameter'][ff_to_ff_throat_indices] = network['pore.diameter'][network.pores('ff')][0]
    
    for i, throat_idx in enumerate(ff_to_ff_throat_indices):
        z1 = network['pore.centroid'][ff_pore_indices[i]][2]
        z2 = network['pore.centroid'][ff_pore_indices[i+1]][2]
        network['throat.length'][throat_idx] = z2 - z1 - network['pore.diameter'][network.pores('ff')][0]
    # Add throat area
    network['throat.area'][ff_to_ff_throat_indices] = np.pi / 4 * network['throat.diameter'][ff_to_ff_throat_indices]**2

    # Add conduit lengths
    for i, throat_idx in enumerate(ff_to_ff_throat_indices):
        pore1_idx = ff_pore_indices[i]
        pore2_idx = ff_pore_indices[i+1]
        network['throat.conduit_lengths.pore1'][throat_idx] = network['pore.diameter'][pore1_idx] / 2
        network['throat.conduit_lengths.throat'][throat_idx] = network['throat.length'][throat_idx]
        network['throat.conduit_lengths.pore2'][throat_idx] = network['pore.diameter'][pore2_idx] / 2
    
    return network

def connect_electrode_to_ff_pores(network: op.network.Network) -> op.network.Network:
    """
    Creates throat connections between FF pores and adjacent channel electrode pores.
    Finds valid electrode pores within a bounding volume (±radius in y and z directions).
    Sets all required throat properties including diameter and length (tip-to-tip distance).
    Labels these throats as 'ff_to_electrode_throat'.
    
    Parameters
    ----------
    network : op.network.Network
        The network with FF pores and electrode pores already defined
    
    Returns
    -------
    op.network.Network
        Network with FF-to-electrode throats added and labeled, with all properties defined.
    """
    ff_pore_indices = network.pores('ff')
    ff_pore_coordinates = network['pore.coords'][network.pores('ff')]
    channel_pores_coords = network['pore.coords'][network.pores('channel')]
    channel_pores_idx = network.pores('channel')
    throat_conns = []
    radius = network['pore.diameter'][network.pores('ff')][0] / 2
    for i, ff_idx in enumerate(ff_pore_indices):
        ff_coord = ff_pore_coordinates[i]
        ff_y, ff_z = ff_coord[1], ff_coord[2]
        
        # Find channel pores within the bounding volume
        # Conditions: y within ±radius, z within ±radius
        y_match = np.abs(channel_pores_coords[:, 1] - ff_y) <= radius
        z_match = np.abs(channel_pores_coords[:, 2] - ff_z) <= radius
        valid_mask = y_match & z_match
        
        # Get the actual pore indices of valid channel pores
        valid_channel_pores = channel_pores_idx[valid_mask]
        
        # Create throat connections [ff_pore, channel_pore]
        for channel_idx in valid_channel_pores:
            throat_conns.append([ff_idx, channel_idx])
    op.topotools.extend(network, conns=throat_conns)
    ff_throat_pore_to_electrode_indices = np.arange(network.Nt - len(throat_conns), network.Nt)
    network.set_label(label='ff_to_electrode_throat', throats=ff_throat_pore_to_electrode_indices)
    network['throat.diameter'][ff_throat_pore_to_electrode_indices] = network['pore.diameter'][network.pores('channel')][0]

    # Calculate throat lengths (tip-to-tip distance)
    for i, throat_idx in enumerate(ff_throat_pore_to_electrode_indices):
        conn = throat_conns[i]  # [ff_pore_idx, channel_pore_idx]
        
        # Get centroids
        ff_centroid = network['pore.centroid'][conn[0]]
        channel_centroid = network['pore.centroid'][conn[1]]
        
        # Get radii
        ff_radius = network['pore.diameter'][conn[0]] / 2
        channel_radius = network['pore.diameter'][conn[1]] / 2
        
        # Calculate centroid-to-centroid distance
        centroid_distance = np.linalg.norm(ff_centroid - channel_centroid)
        
        # Subtract both radii to get tip-to-tip distance
        network['throat.length'][throat_idx] = centroid_distance - ff_radius - channel_radius

    # Add throat area
    network['throat.area'][ff_throat_pore_to_electrode_indices] = np.pi / 4 * network['throat.diameter'][ff_throat_pore_to_electrode_indices]**2

    # Add conduit lengths
    for i, throat_idx in enumerate(ff_throat_pore_to_electrode_indices):
        conn = throat_conns[i]
        network['throat.conduit_lengths.pore1'][throat_idx] = network['pore.diameter'][conn[0]] / 2
        network['throat.conduit_lengths.throat'][throat_idx] = network['throat.length'][throat_idx]
        network['throat.conduit_lengths.pore2'][throat_idx] = network['pore.diameter'][conn[1]] / 2
    return network

def define_membrane(network: op.network.Network) -> op.network.Network:
    """
    Re-labels all pores labeled 'left' to 'membrane'. 'left' pores are opposite to the 'ff' pores.
    This represents the physical structure of a half-stack redox flow battery.
    
    Parameters:
    -----------
    network : OpenPNM Network object
        The network to apply the label change to
    
    Returns:
    --------
    OpenPNM Network object : OpenPNM Network object with the membrane label applied.
    """
    network.set_label(label='membrane', pores=network.pores('left'))
    return network
    
def define_ff_boundaries(network: op.network.Network) -> op.network.Network:
    """
    Labels the first and last FF pores as 'inlet' and 'outlet' respectively.
    
    Parameters:
    -----------
    network : OpenPNM Network object
        The network with FF pores already defined
    
    Returns:
    --------
    OpenPNM Network object : OpenPNM Network object with inlet/outlet labels applied.
    """
    ff_pores = network.pores('ff')
    network.set_label(label='flow_inlet', pores=[ff_pores[0]])
    network.set_label(label='flow_outlet', pores=[ff_pores[-1]])
    return network




