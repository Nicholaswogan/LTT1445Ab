import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
from photochem.extensions import hotrocks
from photochem.utils import stars

from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

import planets
from gridutils import make_grid

def initialize_climate_model(Teq):
    pl = planets.LHS1478b
    st = planets.LHS1478
    c = hotrocks.AdiabatClimateThermalEmission(
        Teq=Teq,
        M_planet=pl.mass,
        R_planet=pl.radius,
        R_star=st.radius,
        Teff=st.Teff,
        metal=st.metal,
        logg=st.logg,
        catdir='sphinx',
        sphinx_filename='data/sphinx.h5'
    )
    c.verbose = False
    return c

def initialize_picaso():

    pl = planets.LHS1478b
    c = initialize_climate_model(pl.Teq)

    filename_db = os.path.join(os.environ['picaso_refdata'],'opacities/')+'opacities_photochem_4.0_25.0_R10000.db'
    ptherm = c.initialize_picaso_from_clima(filename_db)
    c.verbose = False

    wavl = stars.grid_at_resolution(min_wv=np.min(c.ptherm.opa.wave), max_wv=np.max(c.ptherm.opa.wave), R=100)

    return c, ptherm, wavl

def model(x):
    log10PH2O, log10PCO2, log10PO2, log10PSO2, chi, albedo, Teq = x

    p = PICASO

    c = initialize_climate_model(Teq)
  
    # Set albedos
    c.set_custom_albedo(np.array([1.0]), np.array([albedo]))
    p.set_custom_albedo(np.array([1.0]), np.array([albedo]))

    c.chi = chi
    P_i = np.ones(len(c.species_names))*1e-15
    P_i[c.species_names.index('H2O')] = 10.0**log10PH2O
    P_i[c.species_names.index('CO2')] = 10.0**log10PCO2
    P_i[c.species_names.index('O2')] = 10.0**log10PO2
    P_i[c.species_names.index('SO2')] = 10.0**log10PSO2
    P_i *= 1.0e6 # convert to dynes/cm^2

    # Compute climate
    converged = c.RCE_robust(P_i)

    # Save the P-T profile
    P = np.append(c.P_surf,c.P)
    T = np.append(c.T_surf,c.T)

    # Get emission spectra
    _, fp, _ = p.fpfs(c.make_picaso_atm(), wavl=WAVL)

    # Get heat redistribution parameters
    tau_LW, k_term, f_term = c.heat_redistribution_parameters()

    # Save
    result = {
        'converged': np.array(converged),
        'x': x.astype(np.float32),
        'P': P.astype(np.float32),
        'T': T.astype(np.float32),
        'fp': fp.astype(np.float32),
        'f_term': np.array(f_term,np.float32)
    }

    return result

def get_gridvals():
    log10PH2O = np.arange(-4,2.01,1)
    log10PCO2 = np.arange(-7,2.01,0.5)
    log10PO2 = np.arange(-5,2.01,1)
    log10PSO2 = np.arange(-7,-0.99,1)
    chi = np.array([0.05, 0.2, 0.8])
    albedo = np.arange(0,0.401,0.1)
    Teq = np.array([595.0-10*2, 595.0, 595.0+10*2])
    gridvals = (log10PH2O,log10PCO2,log10PO2,log10PSO2,chi,albedo,Teq)
    gridnames = ['log10PH2O','log10PCO2','log10PO2','log10PSO2','chi','albedo','Teq']
    return gridvals, gridnames

NOMINAL_CLIMATE_MODEL, PICASO, WAVL = initialize_picaso()

def main():

    gridvals, gridnames = get_gridvals()
    make_grid(
        model_func=model, 
        gridvals=gridvals,
        gridnames=gridnames, 
        filename='results/LHS1478b_v1.h5', 
        progress_filename='results/LHS1478b_v1.log',
        common={'wavl': WAVL}
    )

if __name__ == "__main__":
    # mpiexec -n 4 python filename.py
    main()
