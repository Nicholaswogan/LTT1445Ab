from photochem.utils import stars

class Star:
    radius : float # relative to the sun
    Teff : float # K
    metal : float # log10(M/H)
    kmag : float
    logg : float
    planets : dict # dictionary of planet objects

    def __init__(self, radius, Teff, metal, kmag, logg, planets):
        self.radius = radius
        self.Teff = Teff
        self.metal = metal
        self.kmag = kmag
        self.logg = logg
        self.planets = planets
        
class Planet:
    radius : float # in Earth radii
    mass : float # in Earth masses
    Teq : float # Equilibrium T in K
    transit_duration : float # in seconds
    eclipse_duration: float # in seconds
    a: float # semi-major axis in AU
    stellar_flux: float # W/m^2
    
    def __init__(self, radius, mass, Teq, transit_duration, eclipse_duration, a, stellar_flux):
        self.radius = radius
        self.mass = mass
        self.Teq = Teq
        self.transit_duration = transit_duration
        self.eclipse_duration = eclipse_duration
        self.a = a
        self.stellar_flux = stellar_flux

# Pass et al. (2023), unless otherwise noted.

LTT1445Ab = Planet(
    radius=1.34,
    mass=2.73,
    Teq=431,
    transit_duration=1.366*60*60,
    eclipse_duration=1.366*60*60, # Assumed same as transit.
    a=0.03810,
    stellar_flux=stars.equilibrium_temperature_inverse(431, 0.0)
)

LTT1445A = Star(
    radius=0.271,
    Teff=3340, # Winters et al. (2022)
    metal=-0.34, # Winters et al. (2022)
    kmag=6.496, # COMPASS JWST_ranking_20200525.xlsx
    logg=4.97, # Exo.Mast
    planets={'b': LTT1445Ab}
)

# Soto et al. (2021)

LHS1478b = Planet(
    radius=1.242,
    mass=2.33,
    Teq=595,
    transit_duration=0.705*60*60,
    eclipse_duration=0.705*60*60, # Assumed same as transit.
    a=0.01848,
    stellar_flux=stars.equilibrium_temperature_inverse(595, 0.0)
)

LHS1478 = Star(
    radius=0.246,
    Teff=3381,
    metal=-0.13,
    kmag=8.8, # Exo.Mast
    logg=4.87,
    planets={'b': LHS1478b}
)