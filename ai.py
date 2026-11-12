import numpy
from config import *
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor



features_risk_prediction = [
    'однородность',
    'влажность_pct',
    'зольность_pct',
    'примеси',
    'размер_частиц_mm',
    'вязкость',
]
model_risk_prediction = CatBoostClassifier()
try:
    model_risk_prediction.load_model(MODELS_DIR / 'risk.cbm')
    print("Загружено из файла")
except:
    df_risk_prediction = pd.read_csv(LEARNING_DIR / 'synthetic_risk_classification.csv')
    X_risk_prediction = df_risk_prediction[features_risk_prediction]
    cat_features = ['однородность', 'примеси', 'вязкость']
    y_risk_prediction = df_risk_prediction['уровень_риска']
    model_risk_prediction.fit(X_risk_prediction, y_risk_prediction, cat_features=cat_features)
    model_risk_prediction.save_model(MODELS_DIR / 'risk.cbm')


# features_energy_prediction = [
#     'влажность_pct',
#     'масса_партии_kg',
#     'температура_карбонизации_C',
#     'время_выдержки_min',
#     'тип_активации',
#     'интенсивность_активации',
#     'размер_частиц_mm',
#     'тип_отхода',
#     'кпд_оборудования',
#     'коэффициент_загрузки',
#     'температура_сушки_C',
#     'время_сушки_min'
# ]
# model_energy_prediction = CatBoostRegressor()
# try:
#     model_energy_prediction.load_model(MODELS_DIR / 'energy.cbm')
#     print("Загружено из файла")
# except:
#     df_energy_prediction = pd.read_csv(LEARNING_DIR / 'synthetic_energy_regression.csv')
#     X_energy_prediction = df_energy_prediction[features_energy_prediction]
#     cat_features = ['тип_активации', 'интенсивность_активации', 'тип_отхода']
#     y_energy_prediction = df_energy_prediction['электрозатраты_kWh']
#     model_energy_prediction.fit(X_energy_prediction, y_energy_prediction, cat_features=cat_features)
#     model_energy_prediction.save_model(MODELS_DIR / 'energy.cbm')


features_activation_prediction = [
    'тип_отхода',
    'источник',
    'влажность_pct',
    'зольность_pct',
    'органическая_часть_pct',
    'примеси',
    'вязкость',
    'размер_частиц_mm',
    'однородность',
    'температура_карбонизации_C',
    'технологический_класс',
    'уровень_риска'
]
model_activation_prediction = CatBoostClassifier()
try:
    model_activation_prediction.load_model(MODELS_DIR / 'activation.cbm')
    print("Загружено из файла")
except:
    df_activation_prediction = pd.read_csv(LEARNING_DIR / 'synthetic_activation_type_classification.csv')
    X_activation_prediction = df_activation_prediction[features_activation_prediction]
    cat_features = ['тип_отхода', 'источник', 'примеси', 'вязкость', 'однородность', 'технологический_класс', 'уровень_риска']
    y_activation_prediction = df_activation_prediction['тип_активации']
    model_activation_prediction.fit(X_activation_prediction, y_activation_prediction, cat_features=cat_features)
    model_activation_prediction.save_model(MODELS_DIR / 'activation.cbm')


features_activation_intensity_prediction = [
    'тип_отхода',
    'источник',
    'влажность_pct',
    'зольность_pct',
    'органическая_часть_pct',
    'примеси',
    'вязкость',
    'размер_частиц_mm',
    'однородность',
    'температура_карбонизации_C',
    'технологический_класс',
    'уровень_риска'
]
model_activation_intensity_prediction = CatBoostClassifier()
try:
    model_activation_intensity_prediction.load_model(MODELS_DIR / 'activation_intensity.cbm')
    print("Загружено из файла")
except:
    df_activation_intensity_prediction = pd.read_csv(LEARNING_DIR / 'synthetic_activation_intensity_classification.csv')
    X_activation_intensity_prediction = df_activation_intensity_prediction[features_activation_intensity_prediction]
    cat_features = ['тип_отхода', 'источник', 'примеси', 'вязкость', 'однородность', 'технологический_класс', 'уровень_риска']
    y_activation_intensity_prediction = df_activation_intensity_prediction['интенсивность_активации']
    model_activation_intensity_prediction.fit(X_activation_intensity_prediction, y_activation_intensity_prediction, cat_features=cat_features)
    model_activation_intensity_prediction.save_model(MODELS_DIR / 'activation_intensity.cbm')


features_batch_classification = [
    'тип_отхода',
    'источник',
    'влажность_pct',
    'зольность_pct',
    'органическая_часть_pct',
    'вязкость',
    'примеси',
    'размер_частиц_mm',
    'однородность'
]
df_batch_classification = pd.read_csv(LEARNING_DIR / 'synthetic_waste_batch_classification.csv')
X_batch_classification = df_batch_classification[features_batch_classification]
cat_features = ['тип_отхода', 'источник', 'вязкость', 'примеси', 'однородность']
model_batch_type = CatBoostClassifier()
try:
    model_batch_type.load_model(MODELS_DIR / 'type.cbm')
    print("Загружено из файла")
except:
    y_batch_type = df_batch_classification['что_это_за_тип_партии']
    model_batch_type.fit(X_batch_classification, y_batch_type, cat_features=cat_features)
    model_batch_type.save_model(MODELS_DIR / 'type.cbm')


model_batch_complexity = CatBoostClassifier()
try:
    model_batch_complexity.load_model(MODELS_DIR / 'complexity.cbm')
    print("Загружено из файла")
except:
    y_batch_complexity = df_batch_classification['насколько_она_сложная']
    model_batch_complexity.fit(X_batch_classification, y_batch_complexity, cat_features=cat_features)
    model_batch_complexity.save_model(MODELS_DIR / 'complexity.cbm')


model_batch_class = CatBoostClassifier()
try:
    model_batch_class.load_model(MODELS_DIR / 'class.cbm')
    print("Загружено из файла")
except:
    y_batch_class = df_batch_classification['какому_технологическому_классу_она_ближе']
    model_batch_class.fit(X_batch_classification, y_batch_class, cat_features=cat_features)
    model_batch_class.save_model(MODELS_DIR / 'class.cbm')


model_batch_preparation = CatBoostClassifier()
try:
    model_batch_preparation.load_model(MODELS_DIR / 'prep.cbm')
    print("Загружено из файла")
except:
    y_batch_preparation = df_batch_classification['нужна_ли_усиленная_подготовка']
    model_batch_preparation.fit(X_batch_classification, y_batch_preparation, cat_features=cat_features)
    model_batch_preparation.save_model(MODELS_DIR / 'prep.cbm')



features_temperature_prediction = [
    'тип_отхода',
    'органическая_часть_pct',
    'зольность_pct',
    'влажность_pct',
    'примеси',
    'размер_частиц_mm',
    'однородность'
]
model_temperature_prediction = CatBoostRegressor()
try:
    model_temperature_prediction.load_model(MODELS_DIR / 'temperature.cbm')
    print("Загружено из файла")
except:
    df_temperature_prediction = pd.read_csv(LEARNING_DIR / 'synthetic_carb_temp_regression.csv')
    X_temperature_prediction = df_temperature_prediction[features_temperature_prediction]
    cat_features = ['тип_отхода', 'примеси', 'однородность']
    y_temperature_prediction = df_temperature_prediction['температура_карбонизации_C']
    model_temperature_prediction.fit(X_temperature_prediction, y_temperature_prediction, cat_features=cat_features)
    model_temperature_prediction.save_model(MODELS_DIR / 'temperature.cbm')



features_time_prediction = [
    'тип_отхода',
    'размер_частиц_mm',
    'однородность',
    'влажность_pct',
    'температура_карбонизации_C',
    'вязкость'
]
model_time_prediction = CatBoostRegressor()
try:
    model_time_prediction.load_model(MODELS_DIR / 'time.cbm')
    print("Загружено из файла")
except:
    df_time_prediction = pd.read_csv(LEARNING_DIR / 'synthetic_hold_time_regression.csv')
    X_time_prediction = df_time_prediction[features_time_prediction]
    cat_features = ['тип_отхода', 'однородность', 'вязкость']
    y_time_prediction = df_time_prediction['время_выдержки_min']
    model_time_prediction.fit(X_time_prediction, y_time_prediction, cat_features=cat_features)
    model_time_prediction.save_model(MODELS_DIR / 'time.cbm')


features_yield_prediction = [
    'органическая_часть_pct',
    'зольность_pct',
    'тип_отхода',
    'влажность_pct',
    'примеси',
    'температура_карбонизации_C',
    'время_выдержки_min',
    'тип_активации',
    'качество_подготовки'
]
model_yield_prediction = CatBoostRegressor()
try:
    model_yield_prediction.load_model(MODELS_DIR / 'yield.cbm')
    print("Загружено из файла")
except:
    df_yield_prediction = pd.read_csv(LEARNING_DIR / 'synthetic_carbon_yield_regression.csv')
    X_yield_prediction = df_yield_prediction[features_yield_prediction]
    cat_features = ['тип_отхода', 'примеси', 'тип_активации', 'качество_подготовки']
    y_yield_prediction = df_yield_prediction['выход_углеродного_материала_pct']
    model_yield_prediction.fit(X_yield_prediction, y_yield_prediction, cat_features=cat_features)
    model_yield_prediction.save_model(MODELS_DIR / 'yield.cbm')


features_drying_temp_prediction = [
    'тип_отхода',
    'источник',
    'влажность_pct',
    'зольность_pct',
    'органическая_часть_pct',
    'примеси',
    'вязкость',
    'размер_частиц_mm',
    'однородность',
    'масса_партии_kg',
    'уровень_риска'
]
model_drying_temp_prediction = CatBoostRegressor()
try:
    model_drying_temp_prediction.load_model(MODELS_DIR / 'drying_temp.cbm')
    print("Загружено из файла")
except:
    df_drying_temp_prediction = pd.read_csv(LEARNING_DIR / 'synthetic_drying_temperature_regression.csv')
    X_drying_temp_prediction = df_drying_temp_prediction[features_drying_temp_prediction]
    cat_features = ['тип_отхода', 'источник', 'примеси', 'вязкость', 'однородность', 'уровень_риска']
    y_drying_temp_prediction = df_drying_temp_prediction['температура_сушки_C']
    model_drying_temp_prediction.fit(X_drying_temp_prediction, y_drying_temp_prediction, cat_features=cat_features)
    model_drying_temp_prediction.save_model(MODELS_DIR / 'drying_temp.cbm')


features_drying_time_prediction = [
    'тип_отхода',
    'источник',
    'влажность_pct',
    'зольность_pct',
    'органическая_часть_pct',
    'примеси',
    'вязкость',
    'размер_частиц_mm',
    'однородность',
    'масса_партии_kg',
    'уровень_риска',
    'температура_сушки_C'
]
model_drying_time_prediction = CatBoostRegressor()
try:
    model_drying_time_prediction.load_model(MODELS_DIR / 'drying_time.cbm')
    print("Загружено из файла")
except:
    df_drying_time_prediction = pd.read_csv(LEARNING_DIR / 'synthetic_drying_time_regression.csv')
    X_drying_time_prediction = df_drying_time_prediction[features_drying_time_prediction]
    cat_features = ['тип_отхода', 'источник', 'примеси', 'вязкость', 'однородность', 'уровень_риска']
    y_drying_time_prediction = df_drying_time_prediction['время_сушки_min']
    model_drying_time_prediction.fit(X_drying_time_prediction, y_drying_time_prediction, cat_features=cat_features)
    model_drying_time_prediction.save_model(MODELS_DIR / 'drying_time.cbm')


print("Обучение завершено!")
