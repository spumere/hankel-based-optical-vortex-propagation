from pyhank import HankelTransform
import numpy as np
import matplotlib.pyplot as plt
from modes import Laguerre_Gaussian
from image_transforms import asm, hasm
from scipy.special import eval_genlaguerre
from math import factorial
from image_transforms import radial_to_cartesian
import time

# Аналитическая формула для пучка Гаусса-Лагерра (Doster T, Watnik AT. Appl Opt 2016; 55: 10239-10246.)
class Laguerre_Gaussian_Exact:
    def __init__(self, N, L, sigma_0, n, m, wavelength, z):
        self.N = N
        self.L = L
        self.sigma_0 = sigma_0
        self.n = n
        self.m = m
        self.wavelength = wavelength
        self.z = z
        # 1. Волновое число k = 2π/λ
        self.k = 2 * np.pi / wavelength
        # 2. Рэлеевская длина z₀ = πσ₀²/λ
        z_0 = np.pi * sigma_0**2 / wavelength
        # 3. Радиус гауссова пучка σ(z) = σ₀√(1 + (z/z₀)²)
        sigma_z = sigma_0 * np.sqrt(1 + (z / z_0)**2)
        # 4. Фаза Гюи ζ(z) = arctan(z / z₀)
        zeta_z = np.arctan(z / z_0)
        # 5. Декартова сетка и переход в цилиндрические координаты (ρ, φ, z)
        x = np.linspace(-L/2, L/2, N, endpoint=False)
        y = np.linspace(-L/2, L/2, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='xy')
        r = np.sqrt(X**2 + Y**2) 
        phi = np.arctan2(Y, X) 
        # 6. Нормировочный коэффициент A_nm = √[2n! / (π(n+|m|)!)]
        A_nm = np.sqrt(2 * factorial(self.n) / (np.pi * factorial(self.n + abs(self.m))))
        # 7. Аргумент для многочлена Лагерра: 2ρ²/σ(z)²
        rho_sigma_squared = 2 * (r**2) / (sigma_z**2)
        # 8. Обобщенный многочлен Лагерра L_n^(|m|)
        L_mn = eval_genlaguerre(self.n, abs(self.m), rho_sigma_squared)
        # 9. Формирование комплексной амплитуды ψ(ρ, φ, z)
        self.complex_amplitude = (
            # 1/σ(z) × A_nm
            (1 / sigma_z) * A_nm *
            # [ (√2ρ/σ(z))^|m| × L_n^|m| × exp(-ρ²/σ(z)²) ]
            ((np.sqrt(2) * r / sigma_z) ** abs(self.m)) * L_mn * np.exp(-r**2 / sigma_z**2) *
            # exp(-i k ρ² z / [2(z² + z₀²)])
            np.exp(-1j * self.k * r**2 * z / (2 * (z**2 + z_0**2))) *
            # exp[i(|m| + 2n + 1)ζ(z)]
            np.exp(1j * (abs(self.m) + 2 * self.n + 1) * zeta_z) *
            # exp(imφ)
            np.exp(1j * self.m * phi)
        )

def get_normalized_data(beam):
    amp = np.abs(beam.complex_amplitude)
    phase = np.angle(beam.complex_amplitude)
    max_val = np.max(amp)
    amp /= max_val
    return amp, phase

def correlation(w, psi):
    return np.abs(np.sum(w * np.conj(psi))) / np.sqrt((
        np.sum(np.abs(w)**2) * np.sum(np.abs(psi)**2)))

# Параметры численного эксперимента
N = 1024
L = 0.03
wavelength = 1550e-9 
sigma_0 = 0.001
z = 5
n, m = 0, 1

exact_beam = Laguerre_Gaussian_Exact(N, L, sigma_0, n, m, wavelength, z)
for_simulation_beam = Laguerre_Gaussian(N, L, sigma_0, n, m) 

start_fourier = time.perf_counter()
propagated_by_fourier = asm(for_simulation_beam, wavelength, z)
end_fourier = time.perf_counter()
time_fourier = end_fourier - start_fourier

r = np.linspace(0, L * np.sqrt(2)/2, (N - 1)//2 + 1, endpoint=True)
transformer = HankelTransform(order=1, radial_grid=r)
start_hankel = time.perf_counter()
propagated_by_hankel = hasm(for_simulation_beam, wavelength, z, transformer)
end_hankel = time.perf_counter()
time_hankel = end_hankel - start_hankel

corr_fourier = correlation(exact_beam.complex_amplitude, propagated_by_fourier.complex_amplitude)
corr_hankel = correlation(exact_beam.complex_amplitude, propagated_by_hankel.complex_amplitude)

print(f"Корреляция (аналитическое решение - ASM): {corr_fourier:.6f}")
print(f"Корреляция (аналитическое решение - HASM): {corr_hankel:.6f}")
print(f"Время выполнения ASM: {time_fourier:.4f} секунд")
print(f"Время выполнения HASM: {time_hankel:.4f} секунд")

fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)

PHASE_MIN = -np.pi
PHASE_MAX = np.pi

# 1. Начальное поле
amp0, ph0 = get_normalized_data(for_simulation_beam)
im_amp0 = axes[0, 0].imshow(amp0, cmap='gray_r', extent=[-L/2, L/2, -L/2, L/2])
axes[0, 0].set_title('Начальное распределение\n(амплитуда)')
im_ph0 = axes[1, 0].imshow(ph0, cmap='gray', extent=[-L/2, L/2, -L/2, L/2], vmin=PHASE_MIN, vmax=PHASE_MAX)
axes[1, 0].set_title('Начальное распределение\n(фаза)')

# 2. Аналитическое решение
amp_exact, ph_exact = get_normalized_data(exact_beam)
im_amp_exact = axes[0, 1].imshow(amp_exact, cmap='gray_r', extent=[-L/2, L/2, -L/2, L/2])
axes[0, 1].set_title(f'Аналитическое решение\n(Δz={z} м, амплитуда)')
im_ph_exact = axes[1, 1].imshow(ph_exact, cmap='gray', extent=[-L/2, L/2, -L/2, L/2], vmin=PHASE_MIN, vmax=PHASE_MAX)
axes[1, 1].set_title(f'Аналитическое решение\n(Δz={z} м, фаза)')

# 3. ASM
amp1, ph1 = get_normalized_data(propagated_by_fourier)
im_amp_fourier = axes[0, 2].imshow(amp1, cmap='gray_r', extent=[-L/2, L/2, -L/2, L/2])
axes[0, 2].set_title(f'Преобразование Фурье\n(амплитуда)')
im_ph_fourier = axes[1, 2].imshow(ph1, cmap='gray', extent=[-L/2, L/2, -L/2, L/2], vmin=PHASE_MIN, vmax=PHASE_MAX)
axes[1, 2].set_title(f'Преобразование Фурье\n(фаза)')

# 4. HASM
amp2, ph2 = get_normalized_data(propagated_by_hankel)
im_amp_hankel = axes[0, 3].imshow(amp2, cmap='gray_r', extent=[-L/2, L/2, -L/2, L/2])
axes[0, 3].set_title(f'Преобразование Ханкеля\n(амплитуда)')
im_ph_hankel = axes[1, 3].imshow(ph2, cmap='gray', extent=[-L/2, L/2, -L/2, L/2], vmin=PHASE_MIN, vmax=PHASE_MAX)
axes[1, 3].set_title(f'Преобразование Ханкеля\n(фаза)')

for ax in axes.flatten():
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')

fig.colorbar(im_amp_hankel, ax=axes[0, 3], shrink=0.8, label='Амплитуда, отн. ед.')

cbar_ph = fig.colorbar(im_ph_hankel, ax=axes[1, 3], shrink=0.8, label='Фаза, рад')
cbar_ph.set_ticks([-np.pi, 0, np.pi])
cbar_ph.set_ticklabels([r'-π', r'0', r'π'])

plt.show()