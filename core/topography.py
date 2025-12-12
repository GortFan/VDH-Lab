import openpnm as op
import numpy as np
import query as q
from  scipy.spatial import cKDTree

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
    
    # Add throat volume (cylinder volume)
    radius = network['throat.diameter'][ff_to_ff_throat_indices] / 2
    network['throat.volume'][ff_to_ff_throat_indices] = np.pi * radius**2 * network['throat.length'][ff_to_ff_throat_indices]
    
    # Add throat centroid (midpoint between connected pores)
    for i, throat_idx in enumerate(ff_to_ff_throat_indices):
        pore1_coord = network['pore.coords'][ff_pore_indices[i]]
        pore2_coord = network['pore.coords'][ff_pore_indices[i+1]]
        network['throat.centroid'][throat_idx] = (pore1_coord + pore2_coord) / 2
    
    # Add throat endpoints (head = pore1 surface, tail = pore2 surface)
    for i, throat_idx in enumerate(ff_to_ff_throat_indices):
        pore1_idx = ff_pore_indices[i]
        pore2_idx = ff_pore_indices[i+1]
        pore1_coord = network['pore.coords'][pore1_idx]
        pore2_coord = network['pore.coords'][pore2_idx]
        direction = (pore2_coord - pore1_coord) / np.linalg.norm(pore2_coord - pore1_coord)
        network['throat.endpoints.head'][throat_idx] = pore1_coord + direction * (network['pore.diameter'][pore1_idx] / 2)
        network['throat.endpoints.tail'][throat_idx] = pore2_coord - direction * (network['pore.diameter'][pore2_idx] / 2)
    
    # Add throat spacing (distance between pore surfaces)
    network['throat.spacing'][ff_to_ff_throat_indices] = network['throat.length'][ff_to_ff_throat_indices]
    
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
    
    # Add throat volume (cylinder volume)
    radius = network['throat.diameter'][ff_throat_pore_to_electrode_indices] / 2
    network['throat.volume'][ff_throat_pore_to_electrode_indices] = np.pi * radius**2 * network['throat.length'][ff_throat_pore_to_electrode_indices]
    
    # Add throat centroid (midpoint between connected pores)
    for i, throat_idx in enumerate(ff_throat_pore_to_electrode_indices):
        conn = throat_conns[i]
        pore1_coord = network['pore.coords'][conn[0]]
        pore2_coord = network['pore.coords'][conn[1]]
        network['throat.centroid'][throat_idx] = (pore1_coord + pore2_coord) / 2
    
    # Add throat endpoints (head = pore1 surface, tail = pore2 surface)
    for i, throat_idx in enumerate(ff_throat_pore_to_electrode_indices):
        conn = throat_conns[i]
        pore1_coord = network['pore.coords'][conn[0]]
        pore2_coord = network['pore.coords'][conn[1]]
        direction = (pore2_coord - pore1_coord) / np.linalg.norm(pore2_coord - pore1_coord)
        network['throat.endpoints.head'][throat_idx] = pore1_coord + direction * (network['pore.diameter'][conn[0]] / 2)
        network['throat.endpoints.tail'][throat_idx] = pore2_coord - direction * (network['pore.diameter'][conn[1]] / 2)
    
    # Add throat spacing (distance between pore surfaces)
    network['throat.spacing'][ff_throat_pore_to_electrode_indices] = network['throat.length'][ff_throat_pore_to_electrode_indices]
    
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

def define_ff_internal_channel_pores(network: op.network.Network) -> op.network.Network:
    """
    Labels all FF pores except the first (inlet) and last (outlet) as 'ff_internal'.
    
    Parameters
    ----------
    network : op.network.Network
        The network with FF pores already defined
    
    Returns
    -------
    op.network.Network
        Network with ff_internal label applied to internal FF pores.
    """
    ff_pores = network.pores('ff')
    internal_ff_pores = ff_pores[1:-1]  # Exclude first and last
    network.set_label(label='ff_internal', pores=internal_ff_pores)
    return network

def create_current_collector_pores(network: op.network.Network) -> op.network.Network:
    ff_pore_radius = network['pore.diameter'][network.pores('ff')][0] / 2
    cc_pore_diameter = 3.74212e-6
    density_factor = 0.1
    
    network.set_label(label='current_collector', pores=network.pores('rib'))
    
    # Reference point and bounds
    ff_center = network['pore.coords'][network.pores('ff')][0]
    x_c, y_c, z_c = ff_center
    z_min = network['pore.coords'][network.pores('ff')][0][2] - ff_pore_radius
    z_max = network['pore.coords'][network.pores('ff')][-1][2] + ff_pore_radius
    
    all_coords = []
    
    # Define 5 faces: (fixed_axis, fixed_value, [vary_axis1, vary_axis2])
    faces = [
        ('x', x_c + ff_pore_radius, 'y', 'z', y_c - ff_pore_radius, y_c + ff_pore_radius, z_min, z_max),  # +x face
        ('y', y_c + ff_pore_radius, 'x', 'z', x_c - ff_pore_radius, x_c + ff_pore_radius, z_min, z_max),  # +y face
        ('y', y_c - ff_pore_radius, 'x', 'z', x_c - ff_pore_radius, x_c + ff_pore_radius, z_min, z_max),  # -y face
        ('z', z_min, 'x', 'y', x_c - ff_pore_radius, x_c + ff_pore_radius, y_c - ff_pore_radius, y_c + ff_pore_radius),  # -z face
        ('z', z_max, 'x', 'y', x_c - ff_pore_radius, x_c + ff_pore_radius, y_c - ff_pore_radius, y_c + ff_pore_radius),  # +z face
    ]
    
    for face in faces:
        fixed_axis, fixed_val, ax1, ax2, min1, max1, min2, max2 = face
        
        n1 = int(((max1 - min1) / cc_pore_diameter) * density_factor)
        n2 = int(((max2 - min2) / cc_pore_diameter) * density_factor)
        
        vals1 = np.linspace(min1, max1, n1)
        vals2 = np.linspace(min2, max2, n2)
        V1, V2 = np.meshgrid(vals1, vals2)
        
        fixed_vals = np.full(V1.size, fixed_val)
        
        # Map to x, y, z based on axis names
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        coords = np.zeros((V1.size, 3))
        coords[:, axis_map[fixed_axis]] = fixed_vals
        coords[:, axis_map[ax1]] = V1.flatten()
        coords[:, axis_map[ax2]] = V2.flatten()
        
        all_coords.append(coords)
    
    cc_pore_coordinates = np.vstack(all_coords)
    
    op.topotools.extend(network=network, coords=cc_pore_coordinates)
    new_cc_pore_indices = np.arange(network.Np - len(cc_pore_coordinates), network.Np)
    network.set_label(label='current_collector', pores=new_cc_pore_indices)
    network['pore.diameter'][new_cc_pore_indices] = cc_pore_diameter
    network['pore.volume'][new_cc_pore_indices] = (4/3) * np.pi * (cc_pore_diameter/2)**3
    network['pore.surface_area'][new_cc_pore_indices] = 4 * np.pi * (cc_pore_diameter/2)**2
    network['pore.area'][new_cc_pore_indices] = np.pi * (cc_pore_diameter/2)**2
    network['pore.centroid'][new_cc_pore_indices] = cc_pore_coordinates

    # Connect current collector pores to FF pores via spatial proximity search
    cc_pore_coords = network['pore.coords'][new_cc_pore_indices]
    ff_pore_coords = network['pore.coords'][network.pores('ff')]
    ff_pore_indices = network.pores('ff')
    throat_conns = []

    # Build KD-tree for efficient spatial queries
    tree = cKDTree(ff_pore_coords)
    search_radius = ff_pore_radius * 2  # 2x to capture corner/edge CC pores (corner pore hypotenouse > radaii by a lot in some cases)

    for i, cc_idx in enumerate(new_cc_pore_indices):
        cc_coord = cc_pore_coords[i]
        
        # Find all FF pores within search radius
        nearby_ff_positions = tree.query_ball_point(cc_coord, r=search_radius)
        
        # Convert positions to actual pore indices
        for pos in nearby_ff_positions:
            ff_idx = ff_pore_indices[pos]
            throat_conns.append([cc_idx, ff_idx])

    op.topotools.extend(network, conns=throat_conns)
    cc_to_ff_throat_indices = np.arange(network.Nt - len(throat_conns), network.Nt)
    network.set_label(label='cc_to_ff_throat', throats=cc_to_ff_throat_indices)
    network['throat.diameter'][cc_to_ff_throat_indices] = cc_pore_diameter

    # Calculate throat lengths (tip-to-tip)
    for i, throat_idx in enumerate(cc_to_ff_throat_indices):
        conn = throat_conns[i]
        cc_centroid = network['pore.coords'][conn[0]]
        ff_centroid = network['pore.coords'][conn[1]]
        
        cc_radius = cc_pore_diameter / 2
        ff_radius = ff_pore_radius
        
        centroid_distance = np.linalg.norm(cc_centroid - ff_centroid)
        network['throat.length'][throat_idx] = centroid_distance - cc_radius - ff_radius

    # Add other required throat properties
    network['throat.area'][cc_to_ff_throat_indices] = np.pi / 4 * network['throat.diameter'][cc_to_ff_throat_indices]**2

    for i, throat_idx in enumerate(cc_to_ff_throat_indices):
        conn = throat_conns[i]
        network['throat.conduit_lengths.pore1'][throat_idx] = cc_pore_diameter / 2
        network['throat.conduit_lengths.throat'][throat_idx] = network['throat.length'][throat_idx]
        network['throat.conduit_lengths.pore2'][throat_idx] = ff_pore_radius

    radius = network['throat.diameter'][cc_to_ff_throat_indices] / 2
    network['throat.volume'][cc_to_ff_throat_indices] = np.pi * radius**2 * network['throat.length'][cc_to_ff_throat_indices]

    for i, throat_idx in enumerate(cc_to_ff_throat_indices):
        conn = throat_conns[i]
        pore1_coord = network['pore.coords'][conn[0]]
        pore2_coord = network['pore.coords'][conn[1]]
        network['throat.centroid'][throat_idx] = (pore1_coord + pore2_coord) / 2
        
        direction = (pore2_coord - pore1_coord) / np.linalg.norm(pore2_coord - pore1_coord)
        network['throat.endpoints.head'][throat_idx] = pore1_coord + direction * (cc_pore_diameter / 2)
        network['throat.endpoints.tail'][throat_idx] = pore2_coord - direction * ff_pore_radius

    network['throat.spacing'][cc_to_ff_throat_indices] = network['throat.length'][cc_to_ff_throat_indices]
    return network

