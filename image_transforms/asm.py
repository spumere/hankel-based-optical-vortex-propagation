"""
[1] Салех Б., Тейх М. Оптика и фотоника. Принципы и применения: Учебное пособие. 
Долгопрудный: Издательский Дом «Интеллект»; 2012. 
[2] Дж. Гудмен. Введение в фурье-оптику. Москва: Мир; 1970.
"""

import numpy as np
from copy import deepcopy
from scipy.fft import fftfreq, fft2, ifft2

def make_propagator(mode, wavelength):
    dx = mode.L/mode.N
    f_x = fftfreq(mode.N, dx)
    f_y = fftfreq(mode.N, dx)
    F_x, F_y = np.meshgrid(f_x, f_y, indexing='xy')
    sqrt_argument = wavelength**(-2) - F_x**2 - F_y**2
    H = np.zeros_like(sqrt_argument, dtype=complex)
    return sqrt_argument, H

def asm(mode, wavelength, z):
    w = deepcopy(mode)
    sqrt_argument, H = make_propagator(w, wavelength)
    
    # Распространяющиеся волны
    propagating_mask = sqrt_argument >= 0
    H[propagating_mask] = np.exp(-1j * 2 * np.pi * z * np.sqrt(sqrt_argument[propagating_mask]))
    
    # Затухающие волны экспоненциально затухают с пройденным расстоянием z, 
    # поэтому ими при z ~ λ можно пренебречь [2]
    evanescent_mask = sqrt_argument < 0
    H[evanescent_mask] = 0
    
    w.complex_amplitude = ifft2(fft2(w.complex_amplitude) * H)

    return w

def iasm(mode, wavelength, z):
    w = deepcopy(mode)
    sqrt_argument, H = make_propagator(w, wavelength)

    # Обратные распространяющиеся волны
    propagating_mask = sqrt_argument >= 0
    H[propagating_mask] = np.exp(1j * 2 * np.pi * z * np.sqrt(sqrt_argument[propagating_mask]))
    
    # Обратные затухающие волны 
    evanescent_mask = sqrt_argument < 0
    H[evanescent_mask] = 0

    w.complex_amplitude = ifft2(fft2(w.complex_amplitude) * H)
    
    return w