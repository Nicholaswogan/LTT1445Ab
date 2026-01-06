from photochem.utils import stars
import json
from pathlib import Path

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


def _load_nea_cache() -> dict:
    cache_path = Path(__file__).resolve().parent / "data" / "nea_planet_cache.json"
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text())


def _system_from_nea_cache(internal_name: str, entry: dict) -> tuple[Planet, Star]:
    pl = entry["planet"]
    st = entry["star"]

    teq = float(pl["teq_k"])
    transit_duration = float(pl["transit_duration_s"])
    planet = Planet(
        radius=float(pl["radius_re"]),
        mass=float(pl["mass_me"]),
        Teq=teq,
        transit_duration=transit_duration,
        eclipse_duration=transit_duration,  # Assumed same as transit.
        a=float(pl["a_au"]),
        stellar_flux=stars.equilibrium_temperature_inverse(teq, 0.0),
    )
    star = Star(
        radius=float(st["radius_rsun"]),
        Teff=float(st["teff_k"]),
        metal=float(st["metal"]),
        kmag=float(st["kmag"]),
        logg=float(st["logg"]),
        planets={},
    )
    return planet, star


SYSTEMS: dict[str, tuple[Planet, Star]] = {
    "LTT1445Ab": (LTT1445Ab, LTT1445A),
    "LHS1478b": (LHS1478b, LHS1478),
}

_nea_cache = _load_nea_cache()
for _internal_name, _entry in _nea_cache.items():
    if _internal_name in SYSTEMS:
        continue
    _planet, _star = _system_from_nea_cache(_internal_name, _entry)
    SYSTEMS[_internal_name] = (_planet, _star)

# Back-fill Star.planets dictionaries for systems defined above.
for _name, (_planet, _star) in SYSTEMS.items():
    _suffix = _name[-1].lower()
    if _suffix.isalpha():
        _star.planets[_suffix] = _planet


# Convenience aliases (variable-safe names) for the NEA-cached systems.
if "TRAPPIST-1b" in SYSTEMS:
    TRAPPIST1b, TRAPPIST1 = SYSTEMS["TRAPPIST-1b"]
if "TRAPPIST-1c" in SYSTEMS:
    TRAPPIST1c, _ = SYSTEMS["TRAPPIST-1c"]
if "LHS1140c" in SYSTEMS:
    LHS1140c, LHS1140 = SYSTEMS["LHS1140c"]
if "GJ357b" in SYSTEMS:
    GJ357b, GJ357 = SYSTEMS["GJ357b"]
if "L9859c" in SYSTEMS:
    L9859c, L98_59 = SYSTEMS["L9859c"]
if "TOI270b" in SYSTEMS:
    TOI270b, TOI270 = SYSTEMS["TOI270b"]
if "GJ3929b" in SYSTEMS:
    GJ3929b, GJ3929 = SYSTEMS["GJ3929b"]
if "GJ1132b" in SYSTEMS:
    GJ1132b, GJ1132 = SYSTEMS["GJ1132b"]
if "TOI1468b" in SYSTEMS:
    TOI1468b, TOI1468 = SYSTEMS["TOI1468b"]
if "GJ486b" in SYSTEMS:
    GJ486b, GJ486 = SYSTEMS["GJ486b"]
if "HD260655b" in SYSTEMS:
    HD260655b, HD260655 = SYSTEMS["HD260655b"]
if "GJ3473b" in SYSTEMS:
    GJ3473b, GJ3473 = SYSTEMS["GJ3473b"]
