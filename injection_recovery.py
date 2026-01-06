import warnings
warnings.filterwarnings('ignore')

import numpy as np
import json
import os
from photochem.utils import stars
from pymultinest.solve import solve
import pickle
import gridutils

import injection_recovery_grid
import planets

_MIRI_ERROR_CACHE_PATH = os.path.join("data", "miri_filter_fpfs_err.json")
try:
    with open(_MIRI_ERROR_CACHE_PATH, "r") as f:
        _MIRI_FILTER_ERRORS = json.load(f)
except FileNotFoundError:
    _MIRI_FILTER_ERRORS = {}

def make_interpolators(filename):
    g = gridutils.GridInterpolator(filename)
    wavl_spectra = g.common['wavl']
    spectra = g.make_interpolator('fpfs')
    return wavl_spectra, spectra

def quantile_to_uniform(quantile, lower_bound, upper_bound):
    return quantile*(upper_bound - lower_bound) + lower_bound

def model_atm_raw(x, wavl, wavl_spectra, spectra):

    log10PCO2 = x[0]

    # Inputs to model grid
    y = np.array([log10PCO2])
    fpfs = spectra(y)
    fpfs = stars.rebin(wavl_spectra, fpfs, wavl)

    return fpfs

def model_atm(x, wv_bins, wavl_spectra, spectra):
    fpfs1 = model_atm_raw(x, wavl_spectra, wavl_spectra, spectra)

    fpfs = np.empty(len(wv_bins))
    for i,b in enumerate(wv_bins):
        fpfs[i] = stars.rebin(wavl_spectra, fpfs1, b)

    return fpfs

def prior_atm(cube):
    params = np.zeros_like(cube)
    params[0] = quantile_to_uniform(cube[0], -7, 2) # log10PCO2
    return params 

def make_loglike(data_dict, wavl_spectra, spectra):
    def loglike(cube):
        data_bins = data_dict['bins']
        y = data_dict['fpfs']
        e = data_dict['err']
        resulty = model_atm(cube, data_bins, wavl_spectra, spectra)
        if np.any(np.isnan(resulty)):
            return -1.0e100 # outside implicit priors
        loglikelihood = -0.5*np.sum((y - resulty)**2/e**2)
        return loglikelihood
    return loglike

def make_loglike_prior(data_dict, wavl_spectra, spectra):

    param_names = ['log10PCO2']
    loglike = make_loglike(data_dict, wavl_spectra, spectra)

    out = {
        'loglike': loglike,
        'prior': prior_atm,
        'param_names': param_names,
        'data_dict': data_dict,
        'wavl_spectra': wavl_spectra,
        'spectra': spectra,
    }

    return out

def sample_log10_psurf(
    mu_bare: float,
    sigma_bare: float,
    mu_atm: float,
    sigma_atm: float,
    v_esc: float,
    f_xuv: float,
    alpha: float,
    s0: float,
    rng: np.random.Generator | None = None,
) -> tuple[float, bool]:
    """
    Sample log10(surface pressure) from a two-regime population model:
      s = v_esc**alpha / f_xuv
      log10(Psurf) ~ Normal(mu_bare, sigma_bare) if s < s0
                   ~ Normal(mu_atm,  sigma_atm)  if s >= s0
    """
    if rng is None:
        rng = np.random.default_rng()

    if s0 <= 0:
        raise ValueError("s0 must be > 0")

    s = (v_esc**alpha) / f_xuv
    is_atm = s >= s0
    mu, sigma = (mu_atm, sigma_atm) if is_atm else (mu_bare, sigma_bare)
    return float(rng.normal(loc=mu, scale=sigma)), bool(is_atm)

def SNR_of_integration(tau, time_per_integration, std_per_integration, extracted_flux):
    num_integrations = max(1, int(tau // time_per_integration))
    std = std_per_integration/np.sqrt(num_integrations)
    SNR = extracted_flux/std
    return SNR

def fpfs_std(SNR_in, SNR_out):
    return np.sqrt((1/SNR_in)**2 + (1/SNR_out)**2)

def get_error(tau_in, tau_out, n_visit, time_per_integration, extracted_flux, std_per_integration):
    SNR_in = SNR_of_integration(tau_in, time_per_integration, std_per_integration, extracted_flux)
    SNR_out = SNR_of_integration(tau_out, time_per_integration, std_per_integration, extracted_flux)
    std = fpfs_std(SNR_in, SNR_out)/np.sqrt(n_visit)
    return std

# Ground-truth population parameters used in `make_data()` for the injected XUV shoreline case.
TRUE_XUV_SHORELINE = {
    "mu_bare": -4.0,
    "sigma_bare": 1.0,
    "mu_atm": 0.0,
    "sigma_atm": 1.0,
    "alpha": 4.0,
    "s0": 140.0,
}

def make_data():
    s0 = TRUE_XUV_SHORELINE["s0"]
    alpha = TRUE_XUV_SHORELINE["alpha"]
    mu_bare = TRUE_XUV_SHORELINE["mu_bare"]
    sigma_bare = TRUE_XUV_SHORELINE["sigma_bare"]
    mu_atm = TRUE_XUV_SHORELINE["mu_atm"]
    sigma_atm = TRUE_XUV_SHORELINE["sigma_atm"]

    np.random.seed(0)
    rng = np.random.default_rng(0)

    data_dicts = {}

    for key in PLANETS:

        # Compute the surface P
        v_esc = PLANETS[key]['pl_vesc'][0]
        f_xuv = PLANETS[key]['pl_log10xuv'][0]
        f_xuv = 10.0**f_xuv
        log10P_surf, is_atm = sample_log10_psurf(
            mu_bare=mu_bare,
            sigma_bare=sigma_bare,
            mu_atm=mu_atm,
            sigma_atm=sigma_atm,
            v_esc=v_esc,
            f_xuv=f_xuv,
            alpha=alpha,
            s0=s0,
            rng=rng
        )
        log10P_surf = np.clip(log10P_surf, -7, 2)
        print(f'{key:<20}{log10P_surf:.1f}  atm_side={is_atm}')

        # Generate some fake data
        data_dict = {
            'bins': np.empty((0,2)),
            'err': np.empty((0,))
        }

        # LRS
        if PLANETS[key]['LRS'] > 0:
            pl, st = planets.SYSTEMS[key]
            c = injection_recovery_grid.initialize_climate_model(pl, st)
            wavl1, fpfs1, fpfs_err1, _ = c.run_pandexo(
                total_observing_time=pl.eclipse_duration*2,
                eclipse_duration=pl.eclipse_duration,
                kmag=st.kmag,
                inst='MIRI LRS',
                calculation='thermal',
                R=None,
                ntrans=PLANETS[key]['LRS']
            )

            # Rebin to 16 bins
            wavl = np.arange(5.0759, 10.61520001, 0.3462)
            _, fpfs_err = stars.rebin_with_errors(wavl1, fpfs1, fpfs_err1, wavl)

            # Get bins
            wv = (wavl[1:] + wavl[:-1])/2
            wv1 = wavl[:-1]
            wv2 = wavl[1:]
            bins = np.empty((len(wv),2))
            for i in range(wv.shape[0]):
                bins[i,:] = np.array([wv1[i],wv2[i]])

            data_dict['bins'] = np.concatenate((data_dict['bins'],bins),axis=0)
            data_dict['err'] = np.append(data_dict['err'],fpfs_err)
        
        if PLANETS[key]['F1280W'][0] > 0:
            bins = np.array([[11.7, 14.0]])
            n_eclipse, time_per_integration, extracted_flux, std_per_integration = PLANETS[key]['F1280W']
            pl, st = planets.SYSTEMS[key]
            tau_in = pl.eclipse_duration
            tau_out = tau_in
            fpfs_err = get_error(tau_in, tau_out, n_eclipse, time_per_integration, extracted_flux, std_per_integration)
            data_dict['bins'] = np.concatenate((data_dict['bins'],bins),axis=0)
            data_dict['err'] = np.append(data_dict['err'],fpfs_err)

        if PLANETS[key]['F1500W'][0] > 0:
            bins = np.array([[13.6, 16.5]])
            n_eclipse, time_per_integration, extracted_flux, std_per_integration = PLANETS[key]['F1500W']
            pl, st = planets.SYSTEMS[key]
            tau_in = pl.eclipse_duration
            tau_out = tau_in
            fpfs_err = get_error(tau_in, tau_out, n_eclipse, time_per_integration, extracted_flux, std_per_integration)
            data_dict['bins'] = np.concatenate((data_dict['bins'],bins),axis=0)
            data_dict['err'] = np.append(data_dict['err'],fpfs_err)

        
        data_dict['wv'] = (data_dict['bins'][:,0] + data_dict['bins'][:,1])/2
        data_dict['wv_err'] = (data_dict['bins'][:,1] - data_dict['bins'][:,0])/2
        
        # Get spectra
        wavl_spectra, spectra = make_interpolators('results/'+key+'_simple.h5')
        x = np.array([log10P_surf]) # CO2
        fpfs = model_atm(x, data_dict['bins'], wavl_spectra, spectra)
        data_dict['fpfs'] = fpfs
        data_dict['truth'] = x
        data_dict['is_atm'] = is_atm

        # Save the data
        data_dicts[key] = data_dict
    
    with open('data/injection_recovery_data.pkl','wb') as f:
        pickle.dump(data_dicts, f)

PLANETS = {
'TRAPPIST-1c': {
    'pl_vesc': (12.207838044539463, 0.27115600503284676), # escape velocity in km/s and 1-sigma error
    'pl_log10xuv': (2.300394944985186, 0.25), # log10 cumulative XUV in Earth XUVs and 1-sigma error
    'pl_log10insol': (0.3469586233262325, 0.020725736303839115), # log10 bolometric insolation Earth insolation and 1-sigma error
    'LRS': 0,
    'F1280W': (4, 41.93, 71892.05, 54.22), # n_eclipse, time-per-integration, extracted-flux, std-per-integration
    'F1500W': (4, 65.59, 59018.17, 37.82) 
    },
'TRAPPIST-1b': {
    'pl_vesc': (12.405077159605309, 0.3197512416490626),
    'pl_log10xuv': (2.573297501254606, 0.25),
    'pl_log10insol': (0.6198611795956526, 0.020850593814826235),
    'LRS': 0,
    'F1280W': (5, 41.93, 71892.05, 54.22),
    'F1500W': (5, 65.59, 59018.17, 37.82)
    },
'LHS1140c': {
    'pl_vesc': (13.699708576437375, 0.25672052549587937),
    'pl_log10xuv': (2.1718335800751793, 0.25),
    'pl_log10insol': (0.7236752426649108, 0.03237495232491283),
    'LRS': 0,
    'F1280W': (0, np.nan, np.nan, np.nan),
    'F1500W': (3, 19.77, 194452.85, 126.16),
    },
'LTT1445Ab': {
    'pl_vesc': (16.579594077424712, 0.8397712564539581),
    'pl_log10xuv': (1.9959349068090098, 0.25),
    'pl_log10insol': (0.7336376828435364, 0.07905486586515731),
    'LRS': 3,
    'F1280W': (0, np.nan, np.nan, np.nan),
    'F1500W': (0, np.nan, np.nan, np.nan),
    },
'GJ357b': {
    'pl_vesc': (13.843837387739072, 1.21646541151659),
    'pl_log10xuv': (2.162877226155182, 0.25),
    'pl_log10insol': (1.0761005676898165, 0.04479669902292471),
    'LRS': 0,
    'F1280W': (0, np.nan, np.nan, np.nan),
    'F1500W': (1, 2.40, 1460598.46, 1173.43),
    },
'L9859c': {
    'pl_vesc': (13.714838825963053, 0.4701786238376248),
    'pl_log10xuv': (2.254585136150937, 0.25),
    'pl_log10insol': (1.1063382786518463, 0.042697714571100276),
    'LRS': 0,
    'F1280W': (0, np.nan, np.nan, np.nan),
    'F1500W': (1, 4.49, 852923.75, 591.04),
    },
'TOI270b': {
    'pl_vesc': (12.796555672277815, 1.0730192342446125),
    'pl_log10xuv': (2.3060846974472478, 0.25),
    'pl_log10insol': (1.2794396524331832, 0.04130353336871573),
    'LRS': 0,
    'F1280W': (0, np.nan, np.nan, np.nan),
    'F1500W': (4, 13.48, 284531.82, 186.19),
    },
'GJ3929b': {
    'pl_vesc': (14.165893624511076, 1.819751054486847),
    'pl_log10xuv': (2.430284939388416, 0.25),
    'pl_log10insol': (1.2799139708156715, 0.0521839260430208),
    'LRS': 0,
    'F1280W': (0, np.nan, np.nan, np.nan),
    'F1500W': (2, 8.99, 420445.98, 280.77),
    },
'GJ1132b': {
    'pl_vesc': (13.881895861964445, 0.7542489918169537),
    'pl_log10xuv': (2.680258032580364, 0.25),
    'pl_log10insol': (1.288344038678611, 0.04584627611766923),
    'LRS': 1,
    'F1280W': (0, np.nan, np.nan, np.nan),
    'F1500W': (0, np.nan, np.nan, np.nan),
    },
 'LHS1478b': {
    'pl_vesc': (15.312826557441337, 0.725893127917154),
    'pl_log10xuv': (2.6076938964481595, 0.25),
    'pl_log10insol': (1.3193406417726623, 0.046572979330732345),
    'LRS': 0,
    'F1280W': (0, np.nan, np.nan, np.nan),
    'F1500W': (2, 20.97, 183869.72, 119.02),
    },
'TOI1468b': {
    'pl_vesc': (16.46857132383645, 1.2941232555856514),
    'pl_log10xuv': (2.6358658203409986, 0.25),
    'pl_log10insol': (1.5606581576767966, 0.03216709452460198),
    'LRS': 0,
    'F1280W': (0, np.nan, np.nan, np.nan),
    'F1500W': (3, 16.17, 235781.76, 154.11),
    },
'GJ486b': {
    'pl_vesc': (16.388968734473007, 0.24408196004867846),
    'pl_log10xuv': (2.755806149503593, 0.25),
    'pl_log10insol': (1.59154782652052, 0.022181680286055272),
    'LRS': 2,
    'F1280W': (0, np.nan, np.nan, np.nan),
    'F1500W': (0, np.nan, np.nan, np.nan),
    },
'HD260655b': {
    'pl_vesc': (14.687038893890502, 1.1746514734307756),
    'pl_log10xuv': (2.4894270821995264, 0.25),
    'pl_log10insol': (1.6255038944404723, 0.010204573184317667),
    'LRS': 0,
    'F1280W': (0, np.nan, np.nan, np.nan),
    'F1500W': (2, 1.5, 2342383.44, 2326.07),
    },
'GJ3473b': {
    'pl_vesc': (13.561918682264043, 1.1254795765758794),
    'pl_log10xuv': (2.8679599235853304, 0.25),
    'pl_log10insol': (1.7732715005854651, 0.049542834900782706),
    'LRS': 0,
    'F1280W': (0, np.nan, np.nan, np.nan),
    'F1500W': (4, 21.57, 179021.76, 115.75),
    }
}

def make_cases():
    with open('data/injection_recovery_data.pkl','rb') as f:
        data_dicts = pickle.load(f)

    cases = {}
    for key in data_dicts:
        wavl_spectra, spectra = make_interpolators('results/'+key+'_simple.h5')
        cases[key] = make_loglike_prior(data_dicts[key], wavl_spectra, spectra)

    return cases

def inference_for_each_planet():
    # Do inference for each atmosphere
    models_to_run = list(RETRIEVAL_CASES.keys())
    for model_name in models_to_run:
        # Setup directories
        outputfiles_basename = f'injection_recovery/{model_name}/{model_name}'
        try:
            os.mkdir(f'injection_recovery/{model_name}')
        except FileExistsError:
            pass
        # Do nested sampling
        results = solve(
            LogLikelihood=RETRIEVAL_CASES[model_name]['loglike'], 
            Prior=RETRIEVAL_CASES[model_name]['prior'], 
            n_dims=len(RETRIEVAL_CASES[model_name]['param_names']), 
            outputfiles_basename=outputfiles_basename, 
            verbose=True,
            n_live_points=1000
        )
        # Save pickle
        pickle.dump(results, open(outputfiles_basename+'.pkl','wb'))

def _logsumexp(logw: np.ndarray, axis: int | None = None) -> np.ndarray:
    logw = np.asarray(logw)
    m = np.max(logw, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(logw - m), axis=axis, keepdims=True))
    if axis is None:
        return out.reshape(())
    return np.squeeze(out, axis=axis)

def _logmeanexp(logw: np.ndarray, axis: int | None = None) -> np.ndarray:
    logw = np.asarray(logw)
    n = logw.shape[axis] if axis is not None else logw.size
    return _logsumexp(logw, axis=axis) - np.log(float(n))

def _normal_logpdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)

def _population_logpdf(
    theta: np.ndarray,
    *,
    mu_bare: float,
    sigma_bare: float,
    mu_atm: float,
    sigma_atm: float,
    alpha: float,
    s0: float,
    v_esc_draws: np.ndarray,
    log10xuv_draws: np.ndarray,
) -> np.ndarray:
    """
    Returns log p(theta | hyperparams) marginalizing over uncertain (v_esc, log10xuv)
    using Monte Carlo draws provided per-target.
    """
    theta = np.asarray(theta, dtype=float)
    v = np.asarray(v_esc_draws, dtype=float)
    log10xuv = np.asarray(log10xuv_draws, dtype=float)

    # Convert log10xuv (Earth units) to linear f_xuv for the s = v^alpha / f relation.
    f_xuv = np.power(10.0, log10xuv)
    s = np.power(v, alpha) / f_xuv
    is_atm = s >= s0

    mu = np.where(is_atm, mu_atm, mu_bare).astype(float)
    sigma = np.where(is_atm, sigma_atm, sigma_bare).astype(float)

    # Broadcast to (n_draws, n_theta)
    logpdf = _normal_logpdf(theta[None, :], mu[:, None], sigma[:, None])
    return _logmeanexp(logpdf, axis=0)

def _population_logpdf_fbol(
    theta: np.ndarray,
    *,
    mu_bare: float,
    sigma_bare: float,
    mu_atm: float,
    sigma_atm: float,
    alpha: float,
    s0: float,
    v_esc_draws: np.ndarray,
    log10insol_draws: np.ndarray,
) -> np.ndarray:
    """
    Like `_population_logpdf`, but uses s = v_esc**alpha / f_bol where f_bol is
    bolometric flux in Earth insolations (f_bol = 10**log10insol).
    """
    theta = np.asarray(theta, dtype=float)
    v = np.asarray(v_esc_draws, dtype=float)
    log10insol = np.asarray(log10insol_draws, dtype=float)

    f_bol = np.power(10.0, log10insol)
    s = np.power(v, alpha) / f_bol
    is_atm = s >= s0

    mu = np.where(is_atm, mu_atm, mu_bare).astype(float)
    sigma = np.where(is_atm, sigma_atm, sigma_bare).astype(float)

    logpdf = _normal_logpdf(theta[None, :], mu[:, None], sigma[:, None])
    return _logmeanexp(logpdf, axis=0)

def inference_hierarchical():
    """
    Run three hierarchical population inferences for `log10PCO2` and write the
    resulting MultiNest `results` dicts (including `logZ`) to disk.

    This function implements the hierarchical reweighting approach described in
    Lustig-Yaeger et al. (2022) for combining per-target posteriors into a
    population-level inference. Each planet has already been “retrieved”
    individually, producing a posterior sample set for `log10PCO2` stored in
    `injection_recovery/<planet>/<planet>.pkl`. Those per-target retrievals used
    a uniform prior on `log10PCO2` over [-7, 2] (see `prior_atm`), which is
    accounted for via importance reweighting.

    What happens here:
    1) Load per-target posterior samples
       - For each planet in `RETRIEVAL_CASES`, load the MultiNest `results` dict
         from `injection_recovery/<planet>/<planet>.pkl` and extract
         `theta = samples[:, 0]`, where `theta` is `log10PCO2`.
       - Deterministically thin each planet’s samples to at most `max_theta`
         to control runtime (seeded RNG so likelihood is reproducible).

    2) Define the per-target “hyper-likelihood” term via posterior reweighting
       - For hyperparameters φ, the marginal likelihood contribution from target i
         is approximated by:

           p(d_i | φ) ∝ E_{θ~p(θ|d_i)}[ p(θ|φ) / p0(θ) ]

         where p0(θ) is the per-target retrieval prior. Numerically this is
         computed as:
           log p(d_i|φ) = logmeanexp( log p(θ_j|φ) - log p0(θ_j) )
         over the posterior samples θ_j.

    3) Fold in uncertain planet properties via Monte Carlo marginalization
       - The `PLANETS` dict provides (mean, 1σ) for:
           - `pl_vesc`      : escape velocity (km/s)
           - `pl_log10xuv`  : log10 cumulative XUV flux (Earth units)
           - `pl_log10insol`: log10 bolometric flux / insolation (Earth units)
       - For models that depend on these properties, we draw `n_vf_draws`
         realizations per planet (seeded RNG) and marginalize the population
         density p(θ|φ, v, …) over those draws using log-mean-exp.
       - These draws are precomputed *outside* the log-likelihood so the
         likelihood is deterministic (important for nested sampling stability).

    4) Run three competing population models (to compare evidences later)
       Model 1: “XUV/escape-velocity threshold” (6 hyperparameters)
         - Population parameters: (mu_bare, sigma_bare, mu_atm, sigma_atm, alpha, s0)
         - Define s = (v_esc**alpha) / f_xuv with f_xuv = 10**(log10xuv)
         - If s < s0: θ ~ Normal(mu_bare, sigma_bare), else θ ~ Normal(mu_atm, sigma_atm)
         - p(θ|φ) is marginalized over uncertain v_esc and log10xuv per target.

       Model 2: “Null single-Gaussian” (2 hyperparameters)
         - θ ~ Normal(mu, sigma) for the whole population, independent of planet properties.

       Model 3: “Bolometric-flux threshold” (6 hyperparameters)
         - Same as Model 1, but uses s = (v_esc**alpha) / f_bol with f_bol = 10**(log10insol)
         - p(θ|φ) is marginalized over uncertain v_esc and log10insol per target.

    5) Save results
       - Each MultiNest run returns a `results` dict from `pymultinest.solve`
         (containing e.g. `logZ`, `samples`, etc.). Each is pickled to:
           - `injection_recovery/hierarchical/model1_s_xuv/hyper.pkl`
           - `injection_recovery/hierarchical/model2_null_gaussian/hyper.pkl`
           - `injection_recovery/hierarchical/model3_s_fbol/hyper.pkl`

    Notes/assumptions:
    - Only the single retrieved parameter `log10PCO2` is used (no multi-D θ).
    - The per-target posterior samples are treated as equally weighted samples
      from p(θ|d_i).
    - The hierarchical likelihood is computed up to a φ-independent constant,
      which is sufficient for evidence ratios/Bayes factors.
    """

    # Loads up posteriors for log10 CO2/total pressure (bar) for each target.
    planet_names = list(RETRIEVAL_CASES.keys())
    posteriors = {}
    for name in planet_names:
        with open(f'injection_recovery/{name}/{name}.pkl','rb') as f:
            result = pickle.load(f)
        samples = result['samples']
        posteriors[name] = samples[:,0]

    # Prior used in the per-target retrieval for log10PCO2 (uniform on [-7, 2]).
    prior_min, prior_max = -7.0, 2.0
    log_p0 = -np.log(prior_max - prior_min)

    # Pre-thin the posterior samples per target (deterministic) to keep runtime reasonable.
    rng = np.random.default_rng(0)
    max_theta = 2000
    theta_by_name: dict[str, np.ndarray] = {}
    for name in planet_names:
        theta_samples = np.asarray(posteriors[name], dtype=float)
        theta_samples = theta_samples[np.isfinite(theta_samples)]
        if theta_samples.size == 0:
            raise ValueError(f"{name}: posterior samples are empty or non-finite")
        if theta_samples.size > max_theta:
            idx = rng.choice(theta_samples.size, size=max_theta, replace=False)
            theta_samples = theta_samples[idx]
        theta_by_name[name] = theta_samples.astype(float)

    # Pre-draw uncertain planet parameters so the hierarchical likelihood is deterministic and fast.
    rng = np.random.default_rng(0)
    n_vf_draws = 128
    vxuv_draws: dict[str, dict[str, np.ndarray]] = {}
    vbol_draws: dict[str, dict[str, np.ndarray]] = {}
    for name in planet_names:
        if name not in PLANETS:
            raise KeyError(f"{name}: missing from PLANETS dict")
        v_mu, v_sig = PLANETS[name]['pl_vesc']
        x_mu, x_sig = PLANETS[name]['pl_log10xuv']
        insol_mu, insol_sig = PLANETS[name]['pl_log10insol']
        v = rng.normal(loc=v_mu, scale=v_sig, size=n_vf_draws)
        v = np.clip(v, 1e-6, None)
        x = rng.normal(loc=x_mu, scale=x_sig, size=n_vf_draws)
        insol = rng.normal(loc=insol_mu, scale=insol_sig, size=n_vf_draws)

        vxuv_draws[name] = {"v_esc": v.astype(float), "log10xuv": x.astype(float)}
        vbol_draws[name] = {"v_esc": v.astype(float), "log10insol": insol.astype(float)}

    def prior_hier_s0alpha(cube):
        params = np.zeros_like(cube)
        params[0] = quantile_to_uniform(cube[0], -7.0, 2.0)   # mu_bare
        params[1] = 10.0 ** quantile_to_uniform(cube[1], -2.0, 1.0)  # sigma_bare (log-uniform)
        params[2] = quantile_to_uniform(cube[2], -7.0, 2.0)   # mu_atm
        params[3] = 10.0 ** quantile_to_uniform(cube[3], -2.0, 1.0)  # sigma_atm (log-uniform)
        params[4] = quantile_to_uniform(cube[4], 0.0, 8.0)    # alpha
        params[5] = quantile_to_uniform(cube[5], -3.0, 6.0)  # log10(s0)
        return params

    def prior_hier_null(cube):
        params = np.zeros_like(cube)
        params[0] = quantile_to_uniform(cube[0], -7.0, 2.0)  # mu
        params[1] = 10.0 ** quantile_to_uniform(cube[1], -2.0, 1.0)  # sigma (log-uniform)
        return params

    def loglike_model1(params):
        mu_bare, sigma_bare, mu_atm, sigma_atm, alpha, log10s0 = params
        s0 = 10.0 ** float(log10s0)
        if sigma_bare <= 0 or sigma_atm <= 0 or s0 <= 0:
            return -1.0e100

        total = 0.0
        for name in planet_names:
            logp = _population_logpdf(
                theta_by_name[name],
                mu_bare=mu_bare,
                sigma_bare=sigma_bare,
                mu_atm=mu_atm,
                sigma_atm=sigma_atm,
                alpha=alpha,
                s0=s0,
                v_esc_draws=vxuv_draws[name]["v_esc"],
                log10xuv_draws=vxuv_draws[name]["log10xuv"],
            )

            # Importance reweighting: p(data|phi) ∝ E_post[ p(theta|phi) / p0(theta) ]
            logw = logp - log_p0
            log_mean_w = _logmeanexp(logw, axis=0)
            if not np.isfinite(log_mean_w):
                return -1.0e100
            total += float(log_mean_w)

        return total

    def loglike_model2_null(params):
        mu, sigma = params
        if sigma <= 0:
            return -1.0e100

        total = 0.0
        for name in planet_names:
            logp = _normal_logpdf(theta_by_name[name], mu, sigma)
            logw = logp - log_p0
            log_mean_w = _logmeanexp(logw, axis=0)
            if not np.isfinite(log_mean_w):
                return -1.0e100
            total += float(log_mean_w)
        return total

    def loglike_model3_fbol(params):
        mu_bare, sigma_bare, mu_atm, sigma_atm, alpha, log10s0 = params
        s0 = 10.0 ** float(log10s0)
        if sigma_bare <= 0 or sigma_atm <= 0 or s0 <= 0:
            return -1.0e100

        total = 0.0
        for name in planet_names:
            logp = _population_logpdf_fbol(
                theta_by_name[name],
                mu_bare=mu_bare,
                sigma_bare=sigma_bare,
                mu_atm=mu_atm,
                sigma_atm=sigma_atm,
                alpha=alpha,
                s0=s0,
                v_esc_draws=vbol_draws[name]["v_esc"],
                log10insol_draws=vbol_draws[name]["log10insol"],
            )
            logw = logp - log_p0
            log_mean_w = _logmeanexp(logw, axis=0)
            if not np.isfinite(log_mean_w):
                return -1.0e100
            total += float(log_mean_w)
        return total

    outdir = "injection_recovery/hierarchical"
    os.makedirs(outdir, exist_ok=True)
    results = {}

    os.makedirs(f"{outdir}/model1_s_xuv", exist_ok=True)
    results["model1_s_xuv"] = solve(
        LogLikelihood=loglike_model1,
        Prior=prior_hier_s0alpha,
        n_dims=6,
        outputfiles_basename=f"{outdir}/model1_s_xuv/hyper",
        verbose=True,
        n_live_points=400,
    )
    pickle.dump(results["model1_s_xuv"], open(f"{outdir}/model1_s_xuv/hyper.pkl", "wb"))

    os.makedirs(f"{outdir}/model2_null_gaussian", exist_ok=True)
    results["model2_null_gaussian"] = solve(
        LogLikelihood=loglike_model2_null,
        Prior=prior_hier_null,
        n_dims=2,
        outputfiles_basename=f"{outdir}/model2_null_gaussian/hyper",
        verbose=True,
        n_live_points=400,
    )
    pickle.dump(results["model2_null_gaussian"], open(f"{outdir}/model2_null_gaussian/hyper.pkl", "wb"))

    os.makedirs(f"{outdir}/model3_s_fbol", exist_ok=True)
    results["model3_s_fbol"] = solve(
        LogLikelihood=loglike_model3_fbol,
        Prior=prior_hier_s0alpha,
        n_dims=6,
        outputfiles_basename=f"{outdir}/model3_s_fbol/hyper",
        verbose=True,
        n_live_points=400,
    )
    pickle.dump(results["model3_s_fbol"], open(f"{outdir}/model3_s_fbol/hyper.pkl", "wb"))

# make_data()
RETRIEVAL_CASES = make_cases()

HIERARCHICAL_PARAMS = {
    'model1_s_xuv': ['mu_bare', 'sigma_bare', 'mu_atm', 'sigma_atm', 'alpha', 'log10s0'],
    'model2_null_gaussian': ['mu', 'sigma'],
    'model3_s_fbol': ['mu_bare', 'sigma_bare', 'mu_atm', 'sigma_atm', 'alpha', 'log10s0'],
}

def make_latex_table():
    """
    Print a LaTeX table summarizing hierarchical model posteriors and evidences.

    Loads the three hierarchical MultiNest results pickles produced by
    `inference_hierarchical()` and prints a `tabular` with:
      - median and 16/84% credible interval for each hyperparameter (when present)
      - log-evidence (`logZ`)
      - Bayes factor relative to Model 1: B_{1X} = exp(logZ_1 - logZ_X)
    """
    results = {}
    for model in ["model1_s_xuv", "model2_null_gaussian", "model3_s_fbol"]:
        with open(f"injection_recovery/hierarchical/{model}/hyper.pkl", "rb") as f:
            results[model] = pickle.load(f)

    logz1 = float(results["model1_s_xuv"]["logZ"])

    def _summary(samples_1d: np.ndarray) -> tuple[float, float, float]:
        q16, q50, q84 = np.percentile(samples_1d, [16.0, 50.0, 84.0])
        return float(q50), float(q50 - q16), float(q84 - q50)

    def _format_pm(med: float, lo: float, hi: float, *, fmt: str = ".1g") -> str:
        return rf"${med:{fmt}}_{{-{lo:{fmt}}}}^{{+{hi:{fmt}}}}$"

    def _format_val(x: float, *, fmt: str = ".1g") -> str:
        return rf"${x:{fmt}}$"

    def _format_sci(x: float) -> str:
        if x == 0 or not np.isfinite(x):
            return r"$0$"
        exp10 = int(np.floor(np.log10(abs(x))))
        mant = x / (10**exp10)
        if -2 <= exp10 <= 2:
            return rf"${x:.2g}$"
        return rf"${mant:.1f}\times 10^{{{exp10}}}$"

    model_labels = {
        "model1_s_xuv": r"$F_\mathrm{XUV}$",
        "model2_null_gaussian": "Null",
        "model3_s_fbol": r"$F_\mathrm{bol}$",
    }

    cols = ["mu_bare", "sigma_bare", "mu_atm", "sigma_atm", "alpha", "log10s0"]
    header = [
        r"\small",
        r"\centering",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lccccccccc}",
        r"\hline",
        r"Model & $\mu_\mathrm{bare}$ & $\sigma_\mathrm{bare}$ & $\mu_\mathrm{atm}$ & $\sigma_\mathrm{atm}$ & $\alpha$ & $\log_{10}s_0$ & $\mathrm{corr}(\alpha,\log_{10}s_0)$ & $\ln Z$ & $B_{1X}$ \\",
        r"\hline",
    ]

    rows = []
    # First row: injected truth (XUV shoreline hyperparameters).
    truth = TRUE_XUV_SHORELINE
    truth_cells = [
        r"True ($F_\mathrm{XUV}$)",
        _format_val(truth["mu_bare"]),
        _format_val(truth["sigma_bare"]),
        _format_val(truth["mu_atm"]),
        _format_val(truth["sigma_atm"]),
        _format_val(truth["alpha"]),
        _format_val(float(np.log10(truth["s0"]))),
        r"--",
        r"--",
        r"--",
    ]
    rows.append(" & ".join(truth_cells) + r" \\")

    # Then fitted models: model1, model3, model2 (null last).
    for model in ["model1_s_xuv", "model3_s_fbol", "model2_null_gaussian"]:
        res = results[model]
        samples = np.asarray(res.get("samples", []), dtype=float)
        logz = float(res["logZ"])

        row_cells = [model_labels.get(model, model)]
        param_names = HIERARCHICAL_PARAMS[model]
        for col in cols:
            if col not in param_names or samples.size == 0:
                row_cells.append(r"--")
                continue
            idx = param_names.index(col)
            med, lo, hi = _summary(samples[:, idx])
            row_cells.append(_format_pm(med, lo, hi))

        # corr(alpha, log10s0) for models that include both parameters
        if samples.size == 0 or ("alpha" not in param_names) or ("log10s0" not in param_names):
            row_cells.append(r"--")
        else:
            a = samples[:, param_names.index("alpha")]
            b = samples[:, param_names.index("log10s0")]
            mask = np.isfinite(a) & np.isfinite(b)
            if np.sum(mask) < 3:
                row_cells.append(r"--")
            else:
                corr = float(np.corrcoef(a[mask], b[mask])[0, 1])
                row_cells.append(_format_val(corr, fmt=".2g"))

        row_cells.append(rf"${logz:.1f}$")
        b1x = float(np.exp(logz1 - logz))
        row_cells.append(r"$1$" if model == "model1_s_xuv" else _format_sci(b1x))
        rows.append(" & ".join(row_cells) + r" \\")

    footer = [r"\hline", r"\end{tabular}"]
    print("\n".join(header + rows + footer))


if __name__ == '__main__':
    # inference_for_each_planet()
    # inference_hierarchical()
    make_latex_table()
