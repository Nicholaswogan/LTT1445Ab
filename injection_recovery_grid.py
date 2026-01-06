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

def initialize_climate_model(pl, st):
    c = hotrocks.AdiabatClimateThermalEmission(
        Teq=pl.Teq,
        M_planet=pl.mass,
        R_planet=pl.radius,
        R_star=st.radius,
        Teff=st.Teff,
        metal=st.metal,
        logg=st.logg,
        catdir='sphinx',
        sphinx_filename='data/sphinx.h5',
        species=['H2O','CO2'], 
        condensates=[]
    )
    c.verbose = False
    return c

def initialize_picaso(pl, st):

    c = initialize_climate_model(pl, st)

    filename_db = os.path.join(os.environ['picaso_refdata'],'opacities/')+'opacities_photochem_4.0_25.0_R10000.db'
    ptherm = c.initialize_picaso_from_clima(filename_db)
    c.verbose = False

    wavl = stars.grid_at_resolution(min_wv=np.min(c.ptherm.opa.wave), max_wv=np.max(c.ptherm.opa.wave), R=100)

    return c, ptherm, wavl

def model(x):
    c = CLIMATE_MODEL
    p = PICASO
    wavl = WAVL

    log10PCO2 = x[0]
    log10PH2O = -4.0

    # Set albedos
    c.set_custom_albedo(np.array([1.0]), np.array([0]))
    p.set_custom_albedo(np.array([1.0]), np.array([0]))

    c.chi =  0.2
    P_i = np.ones(len(c.species_names))*1e-15
    P_i[c.species_names.index('H2O')] = 10.0**log10PH2O
    P_i[c.species_names.index('CO2')] = 10.0**log10PCO2
    P_i *= 1.0e6 # convert to dynes/cm^2

    # Compute climate
    converged = c.RCE_robust(P_i)

    # Save the P-T profile
    P = np.append(c.P_surf, c.P)
    T = np.append(c.T_surf, c.T)

    # Get emission spectra
    _, _, fpfs = p.fpfs(c.make_picaso_atm(), wavl=wavl)

    # Save
    result = {
        'converged': np.array(converged),
        'x': x.astype(np.float32),
        'P': P.astype(np.float32),
        'T': T.astype(np.float32),
        'fpfs': fpfs.astype(np.float32)
    }

    return result

def get_gridvals():
    log10PCO2 = np.arange(-7,2.01,0.5)
    gridvals = (log10PCO2,)
    gridnames = ['log10PCO2']
    return gridvals, gridnames

if __name__ == "__main__":
    gridvals, gridnames = get_gridvals()

    case_names = [
        "TRAPPIST-1c",
        "TRAPPIST-1b",
        "LHS1140c",
        "LTT1445Ab",
        "GJ357b",
        "L9859c",
        "TOI270b",
        "GJ3929b",
        "GJ1132b",
        "LHS1478b",
        "TOI1468b",
        "GJ486b",
        "HD260655b",
        "GJ3473b",
    ]
    cases = {name: planets.SYSTEMS[name] for name in case_names}

    for planet_name in cases:
        pl, st = cases[planet_name]
        CLIMATE_MODEL, PICASO, WAVL = initialize_picaso(pl, st)
        make_grid(
            model_func=model, 
            gridvals=gridvals,
            gridnames=gridnames, 
            filename='results/'+planet_name+'_simple.h5', 
            progress_filename='results/'+planet_name+'_simple.log',
            common={'wavl': WAVL}
        )
