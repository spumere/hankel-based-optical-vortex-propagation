"""
[1] Saleh B, Teich M. Optics and Photonics: Principles and Applications. 
Vol. 1 [In Russian]. Dolgoprudny: Intellekt; 2012. ISBN: 978-5-91559-038-9.
[2] Goodman J. Introduction to Fourier Optics [In Russian]. Moscow: Mir; 1970. 
[3] M Guizar-Sicairos, JC Gutierrez-Vega. Computation of quasi-discrete Hankel 
transforms of integer order for propagating optical wave fields. J. Opt. Soc. Am. A, 2004. 
Vol. 21, No. 1.
"""

import numpy as np
from copy import deepcopy
from image_transforms.radial_to_cartesian import radial_to_cartesian

def hasm(mode, wavelength, z, transformer):
    w = deepcopy(mode)
    k = 2*np.pi / wavelength
    
    sqrt_argument = k**2 - transformer.kr ** 2
    
    H = np.zeros_like(sqrt_argument, dtype=complex)
    
    # Распространяющиеся волны
    propagating_mask = sqrt_argument >= 0
    H[propagating_mask] = np.exp(-1j * z * np.sqrt(sqrt_argument[propagating_mask]))
    
    # Затухающие волны
    evanescent_mask = sqrt_argument < 0
    H[evanescent_mask] = 0

    w.radial_part = transformer.to_original_r(
        transformer.iqdht(
            transformer.qdht(
                transformer.to_transform_r(w.radial_part)) * H))
    w.complex_amplitude = radial_to_cartesian(w.radial_part, w.L,
                                               w.N) * w.azimuthal_part
    return w

def ihasm(mode, wavelength, z, transformer):
    w = deepcopy(mode)
    k = 2*np.pi / wavelength
    
    sqrt_argument = k**2 - transformer.kr ** 2
    
    H = np.zeros_like(sqrt_argument, dtype=complex)
    
    # Обратные распространяющиеся волны
    propagating_mask = sqrt_argument >= 0
    H[propagating_mask] = np.exp(1j * z * np.sqrt(sqrt_argument[propagating_mask]))
    
    # Обратные затухающие волны
    evanescent_mask = sqrt_argument < 0
    H[evanescent_mask] = 0

    w.radial_part = transformer.to_original_r(
        transformer.iqdht(
            transformer.qdht(
                transformer.to_transform_r(w.radial_part)) * H))
    w.complex_amplitude = radial_to_cartesian(w.radial_part, w.L,
                                               w.N) * w.azimuthal_part
    return w