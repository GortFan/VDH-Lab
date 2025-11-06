import openpnm as op
import numpy as np
from scipy import optimize
import transport
# core functions for running various simulations on a fully generated rfb electrode + ff
# constraints: labels must adhere to a pre-defined nomenclature or else nothing will work 

def create_phases(network: op.network.Network, params: dict) -> tuple:
    """
    Creates anolyte and catholyte phases with RFB electrolyte properties.
    
    Uses op.phase.Water as a template class but overrides all transport properties
    with actual electrolyte values from the parameter dictionary. The Water class
    provides the phase structure and methods, but properties are customized for
    the specific TEMPO-based redox flow battery electrolytes.
    
    Parameters
    ----------
    network : op.network.Network
        The OpenPNM network object that these phases will be associated with.
        Must have properly defined pore and throat structures.
    params : dict
        Parameter dictionary containing all physical properties for both electrolytes.
        Required keys:
        - 'anolyte_conductivity' : float [S/m] - Electrical conductivity of anolyte
        - 'anolyte_viscosity' : float [Pa·s] - Dynamic viscosity of anolyte
        - 'anolyte_density' : float [kg/m³] - Mass density of anolyte
        - 'D_a' : float [m²/s] - Active species diffusivity in anolyte
        - 'catholyte_conductivity' : float [S/m] - Electrical conductivity of catholyte
        - 'catholyte_viscosity' : float [Pa·s] - Dynamic viscosity of catholyte
        - 'catholyte_density' : float [kg/m³] - Mass density of catholyte
        - 'D_c' : float [m²/s] - Active species diffusivity in catholyte
    
    Returns
    -------
    tuple[op.phase.Phase, op.phase.Phase]
        A tuple containing (anolyte, catholyte) phase objects with all transport
        properties assigned to every pore in the network.
    
    Notes
    -----
    - All properties are assigned as constant values across all pores
    - Molecular weight is set to 0.01802 kg/mol (water) for both phases
    - These phases can be used directly with OpenPNM transport algorithms
      (StokesFlow, AdvectionDiffusion, OhmicConduction)
    
    Examples
    --------
    >>> from inputDictTEMPO import input_dict as params
    >>> network = op.network.Cubic(shape=[10, 10, 10])
    >>> anolyte, catholyte = create_phases(network, params)
    >>> print(f"Anolyte viscosity: {anolyte['pore.viscosity'][0]:.6f} Pa·s")
    """
    # Create anolyte phase (negative electrode side)
    anolyte = op.phase.Water(network=network, name='anolyte')
    anolyte['pore.electrical_conductivity'] = params['anolyte_conductivity']  # [S/m]
    anolyte['pore.viscosity'] = params['anolyte_viscosity']                   # [Pa·s]
    anolyte['pore.density'] = params['anolyte_density']                       # [kg/m³]
    anolyte['pore.diffusivity'] = params['D_a']                               # [m²/s]
    anolyte['pore.molecular_weight'] = 0.01802                                # [kg/mol]
    
    # Create catholyte phase (positive electrode side)
    catholyte = op.phase.Water(network=network, name='catholyte')
    catholyte['pore.electrical_conductivity'] = params['catholyte_conductivity']  # [S/m]
    catholyte['pore.viscosity'] = params['catholyte_viscosity']                   # [Pa·s]
    catholyte['pore.density'] = params['catholyte_density']                       # [kg/m³]
    catholyte['pore.diffusivity'] = params['D_c']                                 # [m²/s]
    catholyte['pore.molecular_weight'] = 0.01802                                  # [kg/mol]
    
    f_hyd = transport.Flow_shape_factors_ball_and_stick
    anolyte.add_model(propname = 'throat.flow_shape_factors', model = f_hyd)    
    catholyte.add_model(propname = 'throat.flow_shape_factors', model = f_hyd)    
    f_Hyd_cond = transport.Hydraulic_conductance_Hagen_Poiseuille
    anolyte.add_model(propname = 'throat.hydraulic_conductance', model = f_Hyd_cond)
    catholyte.add_model(propname = 'throat.hydraulic_conductance', model = f_Hyd_cond)
    f_poisson = transport.Poisson_shape_factors_ball_and_stick
    anolyte.add_model(propname = 'throat.poisson_shape_factors', model = f_poisson)
    catholyte.add_model(propname = 'throat.poisson_shape_factors', model = f_poisson)
    f_diff_cond = transport.Diffusive_conductance_mixed_diffusion
    anolyte.add_model(propname = 'throat.diffusive_conductance', model = f_diff_cond)
    catholyte.add_model(propname = 'throat.diffusive_conductance', model = f_diff_cond)
    f_ad_dif = transport.Advection_diffusion
    anolyte.add_model(propname = 'throat.ad_dif_conductance', model = f_ad_dif)
    catholyte.add_model(propname = 'throat.ad_dif_conductance', model = f_ad_dif)
    f_elec_con = transport.Electrical_conductance_series_resistors
    anolyte.add_model(propname = 'throat.electrical_conductance', model = f_elec_con)            
    catholyte.add_model(propname = 'throat.electrical_conductance', model = f_elec_con)            
    
    anolyte.regenerate_models()
    catholyte.regenerate_models()     
    return anolyte, catholyte

def setup_stokes_flow(network, phase, Q_inlet, P_outlet=0.0):
    """
    Sets up and solves StokesFlow algorithm for single network.
    Uses 2-stage approach: (1) initial solve with flow rate BC, 
    (2) iterative refinement with pressure BC.
    
    Parameters:
    -----------
    network : OpenPNM Network object
        Network with 'inlet' and 'outlet' labels defined
    phase : OpenPNM Phase object
        Phase with viscosity property defined
    Q_inlet : float
        Total volumetric flow rate at inlet [m³/s]
    P_outlet : float, optional
        Outlet pressure boundary condition [Pa]. Default: 0.0 (atmospheric)
    
    Returns:
    --------
    sf : OpenPNM StokesFlow object
        Solved StokesFlow algorithm with pressure/velocity fields
    
    Notes:
    ------
    Stage 1: Set flow rate BC at inlet, solve to get initial pressure guess
    Stage 2: Use optimize.fsolve to find inlet pressure that matches desired flow rate
    
    Examples:
    ---------
    >>> sf = setup_stokes_flow(network, anolyte, Q_inlet=1e-6, P_outlet=0.0)
    >>> phase.update(sf.soln)
    >>> print(f"Inlet pressure: {phase['pore.pressure'][network.pores('inlet')].mean():.2f} Pa")
    """
    import scipy.optimize as optimize
    import numpy as np
    
    sf = op.algorithms.StokesFlow(network=network, phase=phase)
    
    # Get inlet/outlet pores and throats
    inlet_pores = network.pores('inlet')
    outlet_pores = network.pores('outlet')
    inlet_throats = network.find_neighbor_throats(pores=inlet_pores)
    
    # Calculate cross-sectional areas for flow distribution
    cross_area = np.pi / 4 * network['pore.diameter']**2
    total_inlet_area = np.sum(cross_area[inlet_pores])
    
    # ================== STAGE 1: Initial solve with flow rate BC ==================
    # Distribute flow rate across inlet pores proportional to their area
    for pore in inlet_pores:
        sf.set_rate_BC(
            rates=Q_inlet * cross_area[pore] / total_inlet_area,
            pores=pore
        )
    
    # Set outlet pressure BC
    sf.set_value_BC(values=P_outlet, pores=outlet_pores)
    
    sf.run()
    
    # Update phase and get initial pressure guess
    phase.update(sf.soln)
    P_inlet_guess = phase['pore.pressure'][inlet_pores].mean()
    
    # ================== STAGE 2: Iterative refinement with pressure BC ==================
    # Remove flow rate BCs (switch to pressure BC only)
    sf.clear_rate_BCs()
    
    # Define error function for fsolve
    def pressure_error(P_inlet):
        """
        Calculate relative error between actual and desired flow rate.
        Returns zero when inlet pressure produces exactly Q_inlet.
        """
        # Set inlet pressure BC
        sf.set_value_BC(values=P_inlet, pores=inlet_pores)
        sf.set_value_BC(values=P_outlet, pores=outlet_pores)
        sf.run()
        
        # Update phase and calculate actual flow rate
        phase.update(sf.soln)
        Q_actual = np.sum(phase['throat.flow_rate'][inlet_throats])
        
        # Return relative error (fsolve drives this to zero)
        return (Q_actual - Q_inlet) / Q_inlet
    
    # Find correct inlet pressure using fsolve
    P_inlet_correct = optimize.fsolve(pressure_error, x0=P_inlet_guess)
    
    # ================== FINAL SOLVE: Apply correct pressure and solve ==================
    sf.set_value_BC(values=P_inlet_correct, pores=inlet_pores)
    sf.set_value_BC(values=P_outlet, pores=outlet_pores)
    sf.run()
    
    # Update phase with final solution
    phase.update(sf.soln)
    return sf