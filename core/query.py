import openpnm as op
import numpy as np

def query_throats(network: op.network.Network, label: str = None, property: str = None, index: int = None):
    """
    Query throat properties for troubleshooting and analysis.
    
    Parameters
    ----------
    network : op.network.Network
        The OpenPNM network object to query.
    label : str, optional
        The throat label to filter by (e.g., 'ff_to_electrode_throat', 'ff_to_ff_throat').
        Default is None.
    property : str, optional
        The throat property to query without 'throat.' prefix (e.g., 'length', 'diameter', 'conns').
        Default is None.
    index : int, optional
        Specific index within the filtered throats to query. If None, prints all matching throats.
        Default is None.
    
    Examples
    --------
    >>> query_throats(network, label='ff_to_electrode_throat', property='length')
    >>> query_throats(network, label='ff_to_ff_throat', property='diameter', index=5)
    """
    if index == None:
        print(network[f'throat.{property}'][network.throats(label)])
    else:
        print(network[f'throat.{property}'][network.throats(label)][index])

def query_pores(network: op.network.Network, label: str = None, property: str = None, index: int = None):
    """
    Query pore properties for troubleshooting and analysis.
    
    Parameters
    ----------
    network : op.network.Network
        The OpenPNM network object to query.
    label : str, optional
        The pore label to filter by (e.g., 'ff', 'channel', 'rib', 'left').
        Default is None.
    property : str, optional
        The pore property to query without 'pore.' prefix (e.g., 'diameter', 'coords', 'centroid').
        Default is None.
    index : int, optional
        Specific index within the filtered pores to query. If None, prints all matching pores.
        Default is None.
    
    Examples
    --------
    >>> query_pores(network, label='ff', property='diameter')
    >>> query_pores(network, label='channel', property='coords', index=0)
    """
    if index == None:
        print(network[f'pore.{property}'][network.pores(label)])
    else:
        print(network[f'pore.{property}'][network.pores(label)][index])