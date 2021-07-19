feature_scales = {
    'SPM_A': [0.05, 2.0],
    'SPM_beta': [1.0, 65.0],
    'SPM_chi': [0.0, 2.0],
    'SPM_gamma': [0.0, 0.9],
    'SPM_t0': [-20.0, 20.0],
    'SPM_tau_fall': [5.0, 93.0],
    'SPM_tau_rise': [1.0, 45.0],
    'meanra': [0.0, 360.0],
    'meandec': [-30.0, 90.0],
    'Amplitude': [0.0, 2.4],
    'AndersonDarling': [0.1, 1.0],
    'Autocor_length': [1.0, 20.0],
    'Beyond1Std': [0.05, 0.6],
    'Con': [0.0, 0.13],
    'Eta_e': [8e-4, 4e0],  # Usar log sobre eta_e
    'ExcessVar': [-5e5, 1.4e-3],
    'GP_DRW_sigma': [1e-4, 1.0],  # Usar log sobre gp_drw_sigma
    'GP_DRW_tau': [1e-2, 700.0],  # Usar log sobre gp_drw_tau
    'Gskew': [-1.0, 1.0],
    'Harmonics_mag': [1e-2, 5.0],  # Usar log sobre harmonics_mag
    'Harmonics_mse': [0.0, 0.0140],
    'Harmonics_phase': [0.0, 6.28],
    'IAR_phi': [0.0, 1.0],
    'LinearTrend': [-0.08, 0.08],
    'MHPS_high': [5e-6, 7.0],  # Usar log sobre mhps_high
    'MHPS_low': [1e-4, 30.0],  # clipear con a_min=1e-4 y aplicar log
    'MHPS_ratio': [1e-2, 500.0],  # clipear con a_min=1e-2 y aplicar log
    'MaxSlope': [1.5e-2, 200.0],  # Usar log sobre maxslope
    'Meanvariance': [1.5e-3, 5e-2],  # Usar log sobre meanvariance
    'MedianAbsDev': [1e-2, 1.0],  # Usar log sobre medianabsdev
    'MedianBRP': [0.0, 0.85],
    'Multiband_period': [5e-2, 1e3],  # Usar log sobre multiband_period
    'Period_fit': [0.0, 0.25],
    'delta_period': [3e-8, 5.0],  # clipear y Usar log sobre delta_period
    'PairSlopeTrend': [-0.3, 0.4],
    'PercentAmplitude': [0.0, 0.2],
    'Power_rate': [0.0, 0.9],
    'Psi_CS': [0.0, 0.5],
    'Psi_eta': [0.05, 3.0],
    'Pvar': [1e-5, 1.0],  # Usar log sobre Pvar
    'Q31': [0.05, 1.0],
    'Rcs': [0.1, 0.45],
    'SF_ML_amplitude': [-0.5, 1.7],
    'SF_ML_gamma': [-0.5, 2.0],
    'Skew': [-1.5, 1.5],
    'SmallKurtosis': [-1.7, 6.0],
    'Std': [0.03, 0.6],
    'StetsonK': [0.66, 0.96],
    'delta_mag': [0.05, 2.0],
    'dmag_first_det': [-1.0, 6.5],
    'dmag_non_det': [0.0, 6.5],
    'g-r_max': [-0.5, 2.2],
    'g-r_mean': [-0.3, 2.2],
    'gal_b': [-67.0, 81.6],
    'gal_l': [3.2, 356.8],
    'last_diffmaglim_before': [17.5, 20.9],
    'max_diffmaglim_after': [17.5, 21.0],
    'max_diffmaglim_before': [19.0, 21.0],
    'median_diffmaglim_after': [17.2, 20.5],
    'median_diffmaglim_before': [18.4, 20.8],
    'n_non_det_after': [5.0, 100.0],
    'n_non_det_before': [10.0, 25.0],
    'positive_fraction': [0.0, 1.0],
    'rb': [0.6, 1.0],
    'sgscore1': [0.0, 1.0],
    'W1-W2': [-0.5, 1.4],
    'W2-W3': [-0.3, 4.2],
    'r-W3': [1.5, 10.0],
    'r-W2': [0.9, 8.8],
    'g-W3': [1.9, 11.0],
    'g-W2': [1.0, 10.8]
}

use_log = [
    'Eta_e',
    'GP_DRW_sigma',
    'GP_DRW_tau',
    'Harmonics_mag',
    'MHPS_high',
    'MaxSlope',
    'Meanvariance',
    'MedianAbsDev',
    'Multiband_period',
    'Pvar'
]

use_clip_and_log = [
    'MHPS_low',
    'MHPS_ratio',
    'delta_period'
]
