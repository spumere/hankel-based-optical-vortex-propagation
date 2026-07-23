"""
[1]	Pavelyev VS, Khonina SN, Kotlyar VV, Kazanskiy NL. Control of the 
transverse mode composition of coherent radiation [In Russian]. Samara: 
Samara State Aerospace University Publishing House; 2007. ISBN 978-5-7883-0619-3.
"""

import numpy as np
from scipy.special import eval_genlaguerre
from math import factorial
from image_transforms import radial_to_cartesian

class Laguerre_Gaussian:
    def __init__(self, N, L, sigma_0, n, m):
        self.mode_type = "ГЛ"
        self.N = N
        self.L = L
        self.sigma_0 = sigma_0
        self.n = n
        self.m = m
        # Вычисление радиального сомножителя комплексной амплитуды пучка Гаусса-Лагерра (z=0)
        # 1. Создание сетки полярного радиуса цилиндрической системы координат
        r_radial = np.linspace(0, self.L * np.sqrt(2)/2, (self.N - 1)//2 + 1, endpoint=True)
        # 2. Вычисление многочлена Лагерра L_n^m(2r^2/σ_0^2)
        L_mn = eval_genlaguerre(self.n, abs(self.m), 2 * (r_radial**2) / (self.sigma_0**2))
        # 3. Нормировочная константа
        E_nm = 1/self.sigma_0 * np.sqrt(2*factorial(self.n)/(np.pi*factorial(self.n + abs(self.m))))
        # 4. Радиальный сомножитель
        self.radial_part = E_nm * (np.sqrt(2)*r_radial/self.sigma_0)**abs(self.m) * L_mn * np.exp(
            - r_radial**2/self.sigma_0**2)
        # Вычисление азимутального сомножителя комплексной амплитуды пучка Гаусса-Лагерра
        # 1. Создание координатной сетки в декартовых координатах
        x = np.linspace(-self.L/2, self.L/2, self.N, endpoint=True)
        y = np.linspace(-self.L/2, self.L/2, self.N, endpoint=True)
        X, Y = np.meshgrid(x, y, indexing='xy')
        # 2. Получение азимутального угла φ
        phi = np.arctan2(Y, X)
        # 3. Вычисление азимутальной части exp(imφ)
        self.azimuthal_part = np.exp(1j * self.m * phi)
        # Расчет комплексной амплитуды пучка в начальной плоскости: радиальная часть 
        # преобразуется в двумерное распределение для корректного поэлементного умножения 
        # на азимутальную часть
        self.complex_amplitude = radial_to_cartesian(self.radial_part, self.L, self.N) * self.azimuthal_part