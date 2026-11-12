# -*- coding: utf-8 -*-
from ai import *
import twin

while True:
    data = {}
    print("\n\nРЕКОМЕНДАЦИЯ ПО ПЕРЕРАБОТКЕ\nВведите данные о партии\n")
    data['источник'] = input("""Источник, узел окисления циклогексана
    /участок регенерации ионитов
    /узел ионообменной очистки
    /участок рекуперации смол
    /участок водоподготовки
    /участок обращения с растворами адипатов натрия
    /участок производства капролактама
    /узел нейтрализации адипатных растворов
    /участок концентрирования растворов: """)
    data['тип_отхода'] = input("""Тип отхода, смола окисления циклогексана
    /отработанная ионообменная смола
    /смола растворов адипатов натрия: """)
    data['влажность_pct'] = float(input("Влажность, %: "))
    data['органическая_часть_pct'] = float(input("Органическая часть, %: "))
    data['зольность_pct'] = float(input("Зольность, %: "))
    data['вязкость'] = input("Вязкость, низкая/средняя/высокая: ")
    data['примеси'] = input("Примеси, низкие/умеренные/значимые/повышенные: ")
    data['однородность'] = input("Однородность, низкая/ниже средней/средняя/высокая: ")
    data['размер_частиц_mm'] = float(input("Размер частиц, мм: "))
    data['масса_партии_kg'] = float(input("Масса, кг: "))
    data['макс_масса_kg'] = float(input("Максимальная допустимая масса, кг: "))
    # data['кпд_оборудования'] = float(input("КПД оборудования: "))
    data['кпд_сушильного_узла'] = float(input("КПД сушильного узла: "))
    data['кпд_реактора_карбонизации'] = float(input("КПД реактора карбонизации: "))
    data['коэффициент_загрузки'] = float(data['масса_партии_kg'] / data['макс_масса_kg'])
    data['качество_подготовки'] = input("Качество подготовки, низкое/среднее/высокое: ")

    X_cl = [data[i] for i in features_batch_classification]
    y_type = model_batch_type.predict(X_cl)
    y_complex = model_batch_complexity.predict(X_cl)
    y_techclass = model_batch_class.predict(X_cl)
    y_prep = model_batch_preparation.predict(X_cl)
    if isinstance(y_type, numpy.ndarray):
        y_type = y_type[0]
    if isinstance(y_complex, numpy.ndarray):
        y_complex = y_complex[0]
    if isinstance(y_techclass, numpy.ndarray):
        y_techclass = y_techclass[0]
    if isinstance(y_prep, numpy.ndarray):
        y_prep = y_prep[0]
    data['технологический_класс'] = y_techclass
    data['усиленная_подготовка'] = y_prep

    X_temp = [data[i] for i in features_temperature_prediction]
    y_temp = model_temperature_prediction.predict(X_temp)
    if isinstance(y_temp, numpy.ndarray):
        y_temp = y_temp[0]
    data['температура_карбонизации_C'] = y_temp

    X_time = [data[i] for i in features_time_prediction]
    y_time = model_time_prediction.predict(X_time)
    if isinstance(y_time, numpy.ndarray):
        y_time = y_time[0]
    data['время_выдержки_min'] = y_time

    X_risk = [data[i] for i in features_risk_prediction]
    y_risk = model_risk_prediction.predict(X_risk)
    if isinstance(y_risk, numpy.ndarray):
        y_risk = y_risk[0]
    data['уровень_риска'] = y_risk

    X_activation_type = [data[i] for i in features_activation_prediction]
    y_activation_type = model_activation_prediction.predict(X_activation_type)
    if isinstance(y_activation_type, numpy.ndarray):
        y_activation_type = y_activation_type[0]
    data['тип_активации'] = y_activation_type

    X_activation_intensity = [data[i] for i in features_activation_intensity_prediction]
    y_activation_intensity = model_activation_intensity_prediction.predict(X_activation_intensity)
    if isinstance(y_activation_intensity, numpy.ndarray):
        y_activation_intensity = y_activation_intensity[0]
    data['интенсивность_активации'] = y_activation_intensity

    X_dry_temp = [data[i] for i in features_drying_temp_prediction]
    y_dry_temp = model_drying_temp_prediction.predict(X_dry_temp)
    if isinstance(y_dry_temp, numpy.ndarray):
        y_dry_temp = y_dry_temp[0]
    data['температура_сушки_C'] = y_dry_temp

    X_dry_time = [data[i] for i in features_drying_time_prediction]
    y_dry_time = model_drying_time_prediction.predict(X_dry_time)
    if isinstance(y_dry_time, numpy.ndarray):
        y_dry_time = y_dry_time[0]
    data['время_сушки_min'] = y_dry_time

    # X_energy = [data[i] for i in features_energy_prediction]
    # y_energy = model_energy_prediction.predict(X_energy)
    # if isinstance(y_energy, numpy.ndarray):
    #     y_energy = y_energy[0]
    # data['электрозатраты_kWh'] = y_energy

    X_yield = [data[i] for i in features_yield_prediction]
    y_yield = model_yield_prediction.predict(X_yield)
    if isinstance(y_yield, numpy.ndarray):
        y_yield = y_yield[0]

    twin_output = twin.analysis(W=data['влажность_pct'], W_target=6 if y_prep == "да" else 10,
                                m_batch=data['масса_партии_kg'],
                                m_max=data['макс_масса_kg'], T_dry=y_dry_temp, t_dry=y_dry_time,
                                n_dryer=data['кпд_сушильного_узла'], organic=data['органическая_часть_pct'],
                                viscosity=data['вязкость'], particles=data['размер_частиц_mm'],
                                homogeneity=data['однородность'],
                                imp=data['примеси'], ash=data['зольность_pct'], T_carb=y_temp, t_carb=y_temp,
                                n_reactor=data['кпд_реактора_карбонизации'], Y=y_yield,
                                activation_type=y_activation_type,
                                activation_intensity=y_activation_intensity, risk_pred=y_risk)

    print('\n--РЕКОМЕНДАЦИЯ ПО ПЕРЕРАБОТКЕ--\n')
    print(f'Тип партии: {y_type}\nСложность партии: {y_complex}\nТехнологический класс: {y_techclass}\nУсиленная подготовка: {"обязательна" if y_prep == "да" else "не нужна"}\nТемпература карбонизации: {y_temp}°C\nВремя выдержки: {y_time} мин\nТип активации: {y_activation_type}\nИнтенсивность активации: {y_activation_intensity}\nТемпература сушки: {y_dry_temp}°C\nВремя сушки: {y_dry_time} мин\nМасса воды: {twin_output["m_water"]} кг\nМасса сухого вещества: {twin_output["m_dry"]} кг\nИспарённая вода при сушке: {twin_output["m_evap"]} кг\nОстаточная влажность после сушки: {twin_output["W_residual"]}%\nМасса материала после сушки: {twin_output["m_after_dry"]} кг\nЭнергия на сушку: {twin_output["Q_dry_total_kj"]} кДж\nЭнергия на карбонизация: {twin_output["Q_carb_total_kj"]} кДж\nЭнергия на активацию: {twin_output["Q_act_kj"]} кДж\nСуммарные энергозатраты: {twin_output["E_total_kwh"]} кВтч\nРиск недокарбонизации: {twin_output["I_under"]}\nРиск перегрева: {twin_output["I_over"]}\nИтоговый выход углеродного материала с учётом потерь: {twin_output["Y_final"]}%\nМасса готового продукта: {twin_output["m_product"]} кг\nУдельные энергозатраты: {twin_output["I_energy"]} кВтч/кг\nСтоимость энергии: {twin_output["energy_cost"]} руб\nЭкономический результат партии: {twin_output["revenue"]} руб\nОбщий риск: {y_risk}\nМаксимальная допустимая загрузка: {"превышена ⚠️" if data["масса_партии_kg"] / data["макс_масса_kg"] > 1 else "не превышена"}')
    print('\n------------------------\n\n\n')