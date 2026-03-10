import numpy as np
import matplotlib.pyplot as plt

def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + k1*dt/2)
    k3 = f(t + dt/2, y + k2*dt/2)
    k4 = f(t + dt, y + k3*dt)
    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6

def newton(v0, alpha, material_density=7874, air_density=1.225, C=0.15, dt=0.001, g=9.81, r=0.29):
    S = np.pi * r ** 2
    radius = r
    m = material_density * (4/3) * np.pi * radius**3
    beta = C * air_density * S / 2.0
    u = v0 * np.cos(alpha)
    w = v0 * np.sin(alpha)
    x = 0.0
    y = 0.0
    t = 0.0
    max_t = max(4 * v0 * np.sin(alpha) / g, 30.0)
    x_prev, y_prev = x, y
    trajectory_x = [x]
    trajectory_y = [y]
    while t < max_t:
        state = np.array([x, y, u, w])
        def f(t0, s):
            ux, yy, uu, ww = s
            V = np.sqrt(uu*uu + ww*ww)
            if V < 1e-12:
                du = 0.0
                dw = -g
            else:
                du = -beta * uu * V / m
                dw = -g - beta * ww * V / m
            return np.array([uu, ww, du, dw])
        new = rk4_step(f, t, state, dt)
        x_prev, y_prev = x, y
        x, y, u, w = new
        trajectory_x.append(x)
        trajectory_y.append(y)
        t += dt
        if y < 0:
            if y == y_prev:
                return x, trajectory_x, trajectory_y
            frac = -y_prev / (y - y_prev)
            final_x = x_prev + frac*(x - x_prev)
            trajectory_x[-1] = final_x
            trajectory_y[-1] = 0.0
            return final_x, trajectory_x, trajectory_y
    return x, trajectory_x, trajectory_y

def galilei(v0, alpha, g=9.81):
    range_gal = v0*v0 * np.sin(2*alpha) / g
    t_max = 2 * v0 * np.sin(alpha) / g
    t_array = np.linspace(0, t_max, 1000)
    x_array = v0 * np.cos(alpha) * t_array
    y_array = v0 * np.sin(alpha) * t_array - 0.5 * g * t_array**2
    return range_gal, x_array.tolist(), y_array.tolist()

if __name__ == '__main__':
    v0 = 185
    alpha_deg = 53
    alpha = np.deg2rad(alpha_deg)
    
    range_gal, x_gal, y_gal = galilei(v0, alpha)
    range_newt, x_newt, y_newt = newton(v0, alpha)
    diff = range_gal - range_newt
    
    print(f'Дальность полёта (модель Галилея): {range_gal:.6f} м')
    print(f'Дальность полёта (модель Ньютона): {range_newt:.6f} м')
    print(f'Разница: {diff:.6f} м')
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_gal, y_gal, 'b-', linewidth=2, label='Модель Галилея')
    plt.plot(x_newt, y_newt, 'r-', linewidth=2, label='Модель Ньютона')
    plt.xlabel('Расстояние (м)')
    plt.ylabel('Высота (м)')
    plt.title('Траектории снаряда')
    plt.grid(True)
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()