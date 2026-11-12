from math import log

"""
W - влажность партии
W_target - целевая влажность после сушки, 8-12% без усиленной подготовки, 5-8% с усиленной подготовкой
m_batch - масса партии
m_max - максимальная масса партии
T_dry - температура сушки
t_dry - время сушки
n_dryer - КПД сушильного узла
organic - органическая часть
viscosity - вязкость
particles - размер частиц, мм
homogeneity - однородность
imp - примеси
ash - зольность
T_carb - температура карбонизации
t_carb - время выдержки
n_reactor - КПД реактора карбонизации
Y - выход углеродного материала
activation_type - тип активации
activation_intensity - интенсивность активации
"""

def analysis(W, W_target, m_batch, m_max, T_dry, t_dry, n_dryer, organic, viscosity, particles, homogeneity, imp, ash, T_carb, t_carb, n_reactor, Y, activation_type, activation_intensity, risk_pred):
    m_water = m_batch * (W / 100)
    m_dry = m_batch - m_water
    m_evap = max(0, m_batch * ((W - W_target) / 100))
    Q_evap = m_evap * 2257
    c_p = 1.5
    T_in = 15
    Q_heat = m_batch * c_p * (T_dry - T_in)
    Q_dry_total = (Q_evap + Q_heat) / n_dryer
    T_threshold = 60
    k = 0.03
    f_visc = 1.05 if viscosity == 'низкая' else 1.00 if viscosity == 'средняя' else 0.85
    f_part = 1.1 if particles < 1 else 1.05 if particles < 2 else 1 if particles < 3 else 0.9 if particles < 4 else 0.85
    f_hom = 1.00 if homogeneity == 'высокая' else 0.90 if homogeneity == 'средняя' else 0.85 if homogeneity == 'ниже_средней' else 0.80
    f_imp = 1.00 if imp == 'низкие' else 0.95 if imp == 'умеренные' else 0.85 if imp == 'значимые' else 0.80
    H = f_visc * f_part * f_hom * f_imp
    W_residual = max(W_target, W - k * max(0, T_dry - T_threshold) * log(1 + t_dry) * H)
    m_water_residual = m_dry * W_residual / (100 - W_residual)
    m_after_dry = m_dry + m_water_residual
    Q_carb_heat = m_after_dry * c_p * (T_carb - T_in)
    Q_carb_total = (Q_carb_heat * 1.15) / n_reactor
    T_carb_min_required = 400
    T_carb_safe = 500
    T_carb_max = 800
    fT = min(1, max(0, (T_carb_min_required - T_carb) / T_carb_min_required))
    ft = min(1, max(0, (35 - t_carb) / 35))
    fW = min(1, max(0, (W_residual - W_target) / W_target))
    I_under = 0.4*fT + 0.3*ft + 0.3*fW
    gT = min(1, max(0, (T_carb - T_carb_safe) / (T_carb_max - T_carb_safe)))
    gt = min(1, max(0, (t_carb - 50) / (80 - 50)))
    gorganic = organic / 100
    gmass = min(1, m_batch / m_max)
    I_over = 0.3*gT + 0.3*gt + 0.2*gorganic + 0.2*gmass
    L_activation = 0.5*(1 if activation_intensity == 'низкая' else 2 if activation_intensity == 'средняя' else 3) + 0.3*ash + 0.7*(2 if activation_type == 'пар' else 1 if activation_type == 'CO2' else 0)
    Y_final = max(0, Y - L_activation)
    m_product = m_dry * (Y_final / 100)
    if activation_type == 'CO2':
        q_act_base = 4000
    elif activation_type == 'пар':
        q_act_base = 6000
    else:
        q_act_base = 0
    k_intensity = 1 if activation_intensity == 'низкая' else 1.3 if activation_intensity == 'средняя' else 1.6
    Q_act = m_product * q_act_base * k_intensity
    E_total_kj = Q_dry_total + Q_carb_total + Q_act
    E_total_kwh = E_total_kj / 3600
    I_energy = E_total_kwh / m_product if m_product > 0 else float("inf")
    electricity_price = 6
    energy_cost = E_total_kwh * electricity_price
    price = 100000
    value_replace = 90000
    m_product_t = m_product / 1000
    m_sale = m_product_t * 0.55
    m_internal = m_product_t * 0.45
    revenue = m_sale * price + m_internal * value_replace - energy_cost
    risk_points = 0
    if m_batch / m_max > 1:
        risk_points += 1
    risk_points += 0.75 if risk_pred == 'критический' else 0.5 if risk_pred == 'высокий' else 0.25 if risk_pred == 'средний' else 0
    risk_points += 0.2 * I_over
    risk_points += 0.3 * I_under

    return {
        "m_water": m_water,
        "m_dry": m_dry,
        "m_evap": m_evap,
        "W_residual": W_residual,
        "m_after_dry": m_after_dry,
        "Q_dry_total_kj": Q_dry_total,
        "Q_carb_total_kj": Q_carb_total,
        "Q_act_kj": Q_act,
        "E_total_kwh": E_total_kwh,
        "I_under": I_under,
        "I_over": I_over,
        "Y_final": Y_final,
        "m_product": m_product,
        "I_energy": I_energy,
        "energy_cost": energy_cost,
        "revenue": revenue,
        "risk_points": risk_points
    }
