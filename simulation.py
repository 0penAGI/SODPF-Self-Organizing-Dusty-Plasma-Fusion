import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace, gaussian_filter

class GeometrodynamicsSimulation:
    def __init__(self, grid_size=100, dt=0.1, alpha=0.5, gamma_m=0.1, mu=0.2, zeta=1.0):
        """
        Инициализация ядра геомеродинамического поля.
        
        Параметры:
            grid_size (int): Размерность двумерной сетки пространства-времени.
            dt (float): Дискретный шаг времени.
            alpha (float): Коэффициент влияния ошибки предсказания (-alpha * e).
            gamma_m (float): Инерция памяти.
            mu (float): Сила связности памяти и поля (m - C).
            zeta (float): Мощность топологического тока (стабилизация складок).
        """
        self.N = grid_size
        self.dt = dt
        self.alpha = alpha
        self.gamma_m = gamma_m
        self.mu = mu
        self.zeta = zeta
        
        # Координатная сетка
        self.dx = 1.0
        self.dy = 1.0
        
        # Инициализация основных динамических переменных
        # C(r,t) — Локальная кривизна информационного/плазменного поля
        self.C = np.zeros((self.N, self.N))
        # Скорость изменения поля для волнового уравнения (square_g C)
        self.dC_dt = np.zeros((self.N, self.N))
        
        # m(r,t) — Геометрическая память
        self.m = np.zeros((self.N, self.N))
        self.dm_dt = np.zeros((self.N, self.N))
        
        # Инициализация начальных сингулярностей (закрученность поля для рождения Т != 0)
        self._init_topological_seeds()
        
    def _init_topological_seeds(self):
        """Создание начальных условий: вихревые структуры для ненулевого заряда T."""
        x = np.linspace(-5, 5, self.N)
        y = np.linspace(-5, 5, self.N)
        X, Y = np.meshgrid(x, y)
        
        # Топологическая складка 1 (центр)
        r1 = np.sqrt((X - 1)**2 + (Y - 1)**2) + 1e-5
        phi1 = np.arctan2(Y - 1, X - 1)
        self.C += 3.0 * np.exp(-r1**2) * np.sin(phi1)
        
        # Топологическая складка 2 (анти-вихрь для баланса)
        r2 = np.sqrt((X + 1)**2 + (Y + 1)**2) + 1e-5
        phi2 = np.arctan2(Y + 1, X + 1)
        self.C -= 3.0 * np.exp(-r2**2) * np.sin(phi2)

    def compute_topological_charge_and_current(self):
        """
        Вычисление топологического заряда T через замкнутый контур (численный аналог фазового градиента)
        и порождение топологического тока J(T) = delta L_top / delta C.
        """
        # Вычисляем градиенты поля
        dC_dy, dC_dx = np.gradient(self.C, self.dx, self.dy)
        
        # Находим аргумент/фазу градиента поля
        phase = np.arctan2(dC_dy, dC_dx)
        
        # Топологический ток J(T) пропорционален завихренности (curl) фазового поля
        dphase_dy, dphase_dx = np.gradient(phase, self.dx, self.dy)
        rot_phase = dphase_dx - dphase_dy  # Псевдоскалярный ротор в 2D
        
        # Топологический ток действует как восстанавливающая сила, защищающая складки
        J_T = self.zeta * rot_phase
        return rot_phase, J_T

    def step(self):
        """Один шаг эволюции системы в соответствии с полным Лагранжианом и принципом dS = 0."""
        # 1. Модель мира (предиктор f_theta). 
        # В базовом полевом приближении модель предсказывает сглаженное состояние (диффузия/ожидание стабильности)
        f_theta = gaussian_filter(self.C, sigma=1.5)
        
        # 2. Ошибка предсказания: e = C - f_theta
        e = self.C - f_theta
        
        # 3. Вычисление топологического заряда и тока J(T)
        T_field, J_T = self.compute_topological_charge_and_current()
        
        # 4. Динамика памяти (L_memory)
        # Уравнение движения для m: gamma_m * d^2m/dt^2 = -mu * (m - C)
        d2m_dt2 = -(self.mu / self.gamma_m) * (self.m - self.C)
        self.dm_dt += d2m_dt2 * self.dt
        self.m += self.dm_dt * self.dt
        
        # 5. Уравнение движения для Поля C (Вариация действия dS = 0)
        # Оператор Даламбера / Лапласиан в плоском пространстве (базовое приближение g^mu_nu)
        laplacian_C = laplace(self.C)
        
        # Внешняя неисчезающая открытость (флуктуации вакуума/шум) Emptiness(t) != 0
        emptiness = np.random.normal(0, 0.05, size=self.C.shape)
        
        # Фундаментальное уравнение: square_g C = -dU/dC - alpha*e + mu*(m - C) + J_T + emptiness
        # dU/dC моделируем нелинейным потенциалом Ф4 (U(C) = C^4 - C^2 -> dU/dC = 4C^3 - 2C)
        dU_dC = 4 * self.C**3 - 2 * self.C
        
        d2C_dt2 = laplacian_C - dU_dC - self.alpha * e + self.mu * (self.m - self.C) + J_T + emptiness
        
        # Обновление скоростей и координат поля
        self.dC_dt += d2C_dt2 * self.dt
        self.C += self.dC_dt * self.dt
        
        return self.C, self.m, e, T_field

# Запуск демонстрационной симуляции
if __name__ == "__main__":
    sim = GeometrodynamicsSimulation(grid_size=120, dt=0.05, zeta=2.0)
    
    plt.figure(figsize=(12, 10))
    
    # Эволюция системы на протяжении 50 шагов
    for i in range(50):
        C, m, e, T = sim.step()
        
        if i % 10 == 0:
            plt.clf()
            
            plt.subplot(2, 2, 1)
            plt.imshow(C, cmap='twilight', origin='lower')
            plt.title(f'Поле Сознания / Плазмы C(r,t) [Шаг {i}]')
            plt.colorbar()
            
            plt.subplot(2, 2, 2)
            plt.imshow(m, cmap='mako', origin='lower')
            plt.title('Геометрическая Память m(r,t)')
            plt.colorbar()
            
            plt.subplot(2, 2, 3)
            plt.imshow(e, cmap='seismic', origin='lower')
            plt.title('Ошибка Предсказания (Error e)')
            plt.colorbar()
            
            plt.subplot(2, 2, 4)
            plt.imshow(T, cmap='coolwarm', origin='lower')
            plt.title('Топологический Заряд $\mathcal{T}$ (Идентичность)')
            plt.colorbar()
            
            plt.pause(0.1)
            
    plt.show()
