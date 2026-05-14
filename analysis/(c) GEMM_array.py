
# --Global Exposure Mortality Model (GEMM) for PM2.5 health impact assessment--#
# author: Shiyu Deng
# affiliation: University College London
# email: shiyu.deng.23@ucl.ac.uk
# date: January 2026
#---refer to www.pnas.org/cgi/doi/10.1073/pnas.1803222115 ---#

import numpy as np
import pandas as pd
import os



def age_bracket(age):
    breakpoints = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    for i, bp in enumerate(breakpoints):
        if age < bp:
            return i
    return 11  # age >= 80

GAMMA_PARAMS = {
    'Ischaemic Heart Disease':              (1.9, 12.0, 40.2),
    'Strokes':                              (6.2, 16.7, 23.7),
    'Chronic Obstructive Pulmonary Disease':(6.5,  2.5, 32.0),
    'Lung Cancer':                          (6.2,  9.3, 29.8),
    'Lower Respiratory Infections':         (6.4,  5.7,  8.4),
}

HR_PARAMS = {
    'Ischaemic Heart Disease': [
        (0.5070, 0.02458), (0.4762, 0.02309), (0.4455, 0.02160), (0.4148, 0.02011),
        (0.3841, 0.01862), (0.3533, 0.01713), (0.3226, 0.01564), (0.2919, 0.01415),
        (0.2612, 0.01266), (0.2304, 0.01117), (0.1997, 0.00968), (0.1536, 0.00745),
    ],
    'Strokes': [
        (0.4513, 0.11919), (0.4240, 0.11197), (0.3966, 0.10475), (0.3693, 0.09752),
        (0.3419, 0.09030), (0.3146, 0.08307), (0.2872, 0.07585), (0.2598, 0.06863),
        (0.2325, 0.06190), (0.2051, 0.05418), (0.1778, 0.04695), (0.1368, 0.03611),
    ],
    'Chronic Obstructive Pulmonary Disease': [(0.2510, 0.06762)] * 12,
    'Lung Cancer':                           [(0.2942, 0.06147)] * 12,
    'Lower Respiratory Infections':          [(0.4468, 0.11735)] * 12,
}


BASE_MORTALITY = {
    'Ischaemic Heart Disease': {
        'male':   [v * 0.01 for v in [0.0050, 0.0093, 0.0176, 0.0318, 0.0445, 0.0733, 0.1154, 0.1769, 0.2933, 0.5145, 0.8954, 2.7724]],
        'female': [v * 0.01 for v in [0.0027, 0.0045, 0.0081, 0.0150, 0.0226, 0.0415, 0.0710, 0.1227, 0.2174, 0.4044, 0.7344, 2.1418]],
    },
    'Strokes': {
        'male':   [v * 0.01 for v in [0.0041, 0.0082, 0.0156, 0.0322, 0.0477, 0.0878, 0.1441, 0.2601, 0.4586, 0.9122, 1.6103, 4.2612]],
        'female': [v * 0.01 for v in [0.0014, 0.0025, 0.0050, 0.0115, 0.0203, 0.0433, 0.0761, 0.1421, 0.2730, 0.5599, 1.0058, 2.5671]],
    },
    'Chronic Obstructive Pulmonary Disease': {
        'male':   [v * 0.01 for v in [0.0004, 0.0009, 0.0015, 0.0036, 0.0063, 0.0154, 0.0313, 0.0724, 0.1574, 0.4154, 0.8651, 2.9811]],
        'female': [v * 0.01 for v in [0.0003, 0.0005, 0.0009, 0.0018, 0.0033, 0.0075, 0.0156, 0.0369, 0.0846, 0.2256, 0.4880, 1.6045]],
    },
    'Lung Cancer': {
        'male':   [v * 0.01 for v in [0.0004, 0.0010, 0.0023, 0.0054, 0.0104, 0.0215, 0.0396, 0.0583, 0.0875, 0.1279, 0.1616, 0.2103]],
        'female': [v * 0.01 for v in [0.0003, 0.0006, 0.0014, 0.0032, 0.0049, 0.0093, 0.0153, 0.0229, 0.0340, 0.0496, 0.0643, 0.0806]],
    },
    'Lower Respiratory Infections': {
        'male':   [v * 0.01 for v in [0.0008, 0.0010, 0.0013, 0.0018, 0.0022, 0.0035, 0.0053, 0.0092, 0.0160, 0.0397, 0.0884, 0.4738]],
        'female': [v * 0.01 for v in [0.0004, 0.0004, 0.0006, 0.0008, 0.0009, 0.0015, 0.0025, 0.0046, 0.0091, 0.0227, 0.0519, 0.2693]],
    },
}

# ── cross_region coefficient ──
CROSS_COEF = {
    0: 1.1323,  # 25–29
    1: 0.8160,  # 30–39  (注：原代码 max 用 0.8169，mean/min 用 0.8160，统一为 0.8160)
    2: 0.8160,
    3: 0.9465,  # 40–49
    4: 0.9465,
    5: 1.1176,  # 50–59
    6: 1.1176,
    7: 0.9874,  # 60+
    8: 0.9874,
    9: 0.9874,
    10: 0.9874,
    11: 0.9874,
}



def calculate_z(PM2_5):
    return max(0.0, PM2_5 - 2.4)

def gamma_function(PM2_5, disease):
    alpha, mu, pi = GAMMA_PARAMS[disease]
    z = calculate_z(PM2_5)
    return np.log(1 + z / alpha) / (1 + np.exp((mu - z) / pi))

def calculate_hazard_ratio(PM2_5, age, disease):
    gamma = gamma_function(PM2_5, disease)
    theta, se = HR_PARAMS[disease][age_bracket(age)]
    mean  = theta * gamma
    upper = (theta + 1.96 * se) * gamma
    lower = (theta - 1.96 * se) * gamma
    return np.exp(upper), np.exp(lower), np.exp(mean)

def set_base_mortality(age, sex, disease):
    return BASE_MORTALITY[disease][sex][age_bracket(age)]

def PM_mortality(PM2_5, disease, age, sex, population):
    hr_max, hr_min, hr_mean = calculate_hazard_ratio(PM2_5, age, disease)
    base = set_base_mortality(age, sex, disease)
    calc = lambda hr: (1 - 1 / hr) * base * population
    return calc(hr_max), calc(hr_min), calc(hr_mean)

def cross_region(mortality_max, mortality_min, mortality_mean, age, mobility):
    coef = CROSS_COEF[age_bracket(age)]
    factor = mobility * coef
    cap = min(factor * 1.05, 1.0)
    cross_max  = mortality_max  * (cap if factor * 1.05 <= 1 else 1.0)
    cross_mean = mortality_mean * factor
    cross_min  = mortality_min  * factor
    return cross_max, cross_min, cross_mean
 
 
import pandas as pd
import numpy as np
import itertools

diseases = ['Ischaemic Heart Disease', 'Strokes', 'Chronic Obstructive Pulmonary Disease', 'Lung Cancer', 'Lower Respiratory Infections']
air_scenarios = ['ref','cleanair','earlypeak', 'ontimepeak_CL', 'ontimepeak_NZ_CL']
fer_scenarios = ['low','mid','high']
rcp_scenarios = ['RCP2_6', 'RCP4_5', 'RCP8_5']
dev_scenarios = ['SSPFer1_SSPMigr1', 'SSPFer1_SSPMigr2', 'SSPFer1_SSPMigr3', 'SSPFer2_SSPMigr1', 'SSPFer2_SSPMigr2', 'SSPFer2_SSPMigr3',
                  'SSPFer3_SSPMigr1', 'SSPFer3_SSPMigr2', 'SSPFer3_SSPMigr3','SSPFer4_SSPMigr1', 'SSPFer4_SSPMigr2', 'SSPFer4_SSPMigr3',
                  'SSPFer5_SSPMigr1', 'SSPFer5_SSPMigr2', 'SSPFer5_SSPMigr3']

years = list(range(2030, 2061, 10))

all_tasks = list(itertools.product(air_scenarios, fer_scenarios, rcp_scenarios, years, dev_scenarios))

task_id = int(os.environ.get("SGE_TASK_ID", "1")) - 1  # 0-based
air, fer, rcp, year, dev = all_tasks[task_id]


city_file_path ='/city_names.csv' # city names data path
city_data=pd.read_csv(city_file_path)

mobility_file_path='/data source/mobility_rate.csv' # mobility data path
mobility_data=pd.read_csv(mobility_file_path)

mortality = []          

weighted_pm25_path = rf'/{air}_{fer}_{rcp}_{year}.csv' # Weighted PM2.5 data path
weighted_pm25_data = pd.read_csv(weighted_pm25_path)    
                                           
for city in city_data['English']:                                   
    pm25_row = weighted_pm25_data.loc[weighted_pm25_data['city'] == city]
    if pm25_row.empty:
        PM2_5 = np.nan
        continue
    PM2_5 = pm25_row.iloc[0]['weighted_pm25']
                    
    pop_file_path = rf'/pop_city/{city}_{dev}.xlsx' # Population data path
    
    population_data_male = pd.read_excel(pop_file_path, sheet_name='male')
    population_data_female = pd.read_excel(pop_file_path, sheet_name='female')

                                                        
    if city in mobility_data['City'].values:
        mobility = mobility_data.loc[mobility_data['City']==city, 'proportion'].values[0]
    else:
        mobility = np.nan
        
    for disease in diseases:
        for age in range(25, 101):

            pop_male   = population_data_male.loc[population_data_male['Age']==age, year].values[0]
            pop_female = population_data_female.loc[population_data_female['Age']==age, year].values[0]

            mortality_max_male, mortality_min_male, mortality_mean_male= PM_mortality(PM2_5, disease, age,'male', pop_male)
            mortality_max_female, mortality_min_female, mortality_mean_female = PM_mortality(PM2_5, disease, age,'female', pop_female)                        
                            
            mo_total = mortality_mean_male+mortality_mean_female
                                            
            hospital_cost_max_male, hospital_cost_min_male,hospital_cost_mean_male = hospital_cost(disease, mortality_max_male,mortality_min_male,mortality_mean_male,age)  
            hospital_cost_max_female, hospital_cost_min_female,hospital_cost_mean_female = hospital_cost(disease, mortality_max_female,mortality_min_female,mortality_mean_female,age)                  
            hospital_cost_total=(hospital_cost_mean_male+hospital_cost_mean_female)/1000000 # Convert to million
                    
            if not np.isnan(mobility):                
                cross_patient_max_male, cross_patient_min_male,cross_patient_mean_male = cross_region(mortality_max_male, mortality_min_male,mortality_mean_male,age,mobility)  
                cross_patient_max_female, cross_patient_min_female,cross_patient_mean_female = cross_region(mortality_max_female, mortality_min_female,mortality_mean_female,age,mobility) 
                cross_patient_total=(cross_patient_mean_male+cross_patient_mean_female)# Convert to million
            else:
                                
                cross_patient_max_male = cross_patient_min_male = cross_patient_mean_male = np.nan
                cross_patient_max_female = cross_patient_min_female = cross_patient_mean_female = np.nan
                cross_patient_total = np.nan
                            
            mortality.append({
                                'air_scenario':air,
                                'edu_scenario': fer,
                                'rcp_scenario': rcp,
                                'fermig_scenario': dev,                            
                                'disease': disease,                  
                                'city': city,
                                'year': year,
                                'age': age,
                                'pop_male':pop_male,                                             
                                'mo_max_male': mortality_max_male,
                                'mo_min_male': mortality_min_male,
                                'mo_mean_male': mortality_mean_male,                            
                                'hospital_cost_max_male': hospital_cost_max_male,
                                'hospital_cost_min_male': hospital_cost_min_male,
                                'hospital_cost_mean_male': hospital_cost_mean_male,                            
                                'cross_patient_max_male': cross_patient_max_male,
                                'cross_patient_min_male': cross_patient_min_male,  
                                'cross_patient_mean_male': cross_patient_mean_male,                   
                                'pop_female':pop_female,                                              
                                'mo_max_female': mortality_max_female,
                                'mo_min_female': mortality_min_female,
                                'mo_mean_female': mortality_mean_female,
                                'hospital_cost_max_female': hospital_cost_max_female,
                                'hospital_cost_min_female': hospital_cost_min_female,  
                                'hospital_cost_mean_female': hospital_cost_mean_female,                             
                                'cross_patient_max_female': cross_patient_max_female,
                                'cross_patient_min_female': cross_patient_min_female,
                                'cross_patient_mean_female': cross_patient_mean_female,
                                'pop':pop_female+pop_male,                                                                                                                                 
                                'mo_total':mo_total,
                                'hospital_cost_total(million)':hospital_cost_total,
                                'cross_patient_total': cross_patient_total                         
                            })  
outdir = "health_burden"
os.makedirs(outdir, exist_ok=True)
outfile = f"{outdir}/{air}_{fer}_{rcp}_{dev}_{year}.csv"
pd.DataFrame(mortality).to_csv(outfile, index=False)
print(f"[INFO] Saved {outfile}")                                                          
               
import pandas as pd
import numpy as np
import itertools

diseases = ['Ischaemic Heart Disease', 'Strokes', 'Chronic Obstructive Pulmonary Disease', 'Lung Cancer', 'Lower Respiratory Infections']
air_scenarios = ['ref','clean_air','earlypeak_NZ_CL', 'ontimepeak_CL', 'ontimepeak_NZ_CL']
fer_scenarios = ['low','mid','high']
rcp_scenarios = ['RCP2_6', 'RCP4_5', 'RCP8_5']
dev_scenarios = ['SSPFer1_SSPMigr1', 'SSPFer1_SSPMigr2', 'SSPFer1_SSPMigr3', 'SSPFer2_SSPMigr1', 'SSPFer2_SSPMigr2', 'SSPFer2_SSPMigr3',
                  'SSPFer3_SSPMigr1', 'SSPFer3_SSPMigr2', 'SSPFer3_SSPMigr3','SSPFer4_SSPMigr1', 'SSPFer4_SSPMigr2', 'SSPFer4_SSPMigr3',
                  'SSPFer5_SSPMigr1', 'SSPFer5_SSPMigr2', 'SSPFer5_SSPMigr3']

years = list(range(2020, 2061, 10))

city_file_path ='/Users/shirley/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Desktop/city_shp/city_names.csv'
city_data=pd.read_csv(city_file_path)

mobility_file_path='/Users/shirley/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Desktop/air_pollution/data source/mobility_rate.csv'
mobility_data=pd.read_csv(mobility_file_path)
outdir = "/Volumes/UCL/论文工作/空气污染/health_burden/earlypeak_NZ_CL"
os.makedirs(outdir, exist_ok=True)

mobility_lookup = dict(zip(mobility_data['City'], mobility_data['proportion']))


for air, fer, rcp, year in itertools.product(air_scenarios, fer_scenarios, rcp_scenarios, years):

    weighted_pm25_path = rf'/Volumes/UCL/论文工作/空气污染/weighted_pm25/{air}_{fer}_{rcp}_{year}.csv'
    weighted_pm25_data = pd.read_csv(weighted_pm25_path)

    pm25_lookup = weighted_pm25_data.set_index('city')['weighted_pm25'].to_dict()

    for dev in dev_scenarios:
        outfile = f"{outdir}/{air}_{fer}_{rcp}_{dev}_{year}.csv"

        # ✅ skip 已完成的文件
        if os.path.exists(outfile):
            print(f"[SKIP] {outfile}")
            continue

        mortality = []

        for city in city_data['English']:
            PM2_5 = pm25_lookup.get(city, np.nan)
            if np.isnan(PM2_5):
                continue

            pop_file_path = rf'/Volumes/UCL/论文工作/空气污染/pop_city_revised/{city}_{dev}.xlsx'
            population_data_male   = pd.read_excel(pop_file_path, sheet_name='male')
            population_data_female = pd.read_excel(pop_file_path, sheet_name='female')


            pop_male_lookup   = population_data_male.set_index('Age')[year].to_dict()
            pop_female_lookup = population_data_female.set_index('Age')[year].to_dict()

            mobility = mobility_lookup.get(city, np.nan)
            has_mobility = not np.isnan(mobility)

            for disease in diseases:
                for age in range(25, 101):
                    pop_male   = pop_male_lookup.get(age, np.nan)
                    pop_female = pop_female_lookup.get(age, np.nan)

                    mortality_max_male, mortality_min_male, mortality_mean_male = PM_mortality(PM2_5, disease, age, 'male', pop_male)
                    mortality_max_female, mortality_min_female, mortality_mean_female = PM_mortality(PM2_5, disease, age, 'female', pop_female)

                    mo_total     = mortality_mean_male + mortality_mean_female
                    mo_total_min = mortality_min_male  + mortality_min_female
                    mo_total_max = mortality_max_male  + mortality_max_female

                    if has_mobility:
                        cross_patient_max_male, cross_patient_min_male, cross_patient_mean_male     = cross_region(mortality_max_male,   mortality_min_male,   mortality_mean_male,   age, mobility)
                        cross_patient_max_female, cross_patient_min_female, cross_patient_mean_female = cross_region(mortality_max_female, mortality_min_female, mortality_mean_female, age, mobility)
                        cross_patient_total     = cross_patient_mean_male + cross_patient_mean_female
                        cross_patient_total_min = cross_patient_min_male  + cross_patient_min_female
                        cross_patient_total_max = cross_patient_max_male  + cross_patient_max_female
                    else:
                        cross_patient_max_male = cross_patient_min_male = cross_patient_mean_male     = np.nan
                        cross_patient_max_female = cross_patient_min_female = cross_patient_mean_female = np.nan
                        cross_patient_total = cross_patient_total_min = cross_patient_total_max        = np.nan

                    mortality.append({
                        'air_scenario':    air,
                        'edu_scenario':    fer,
                        'rcp_scenario':    rcp,
                        'fermig_scenario': dev,
                        'disease':         disease,
                        'city':            city,
                        'year':            year,
                        'age':             age,
                        'pop_male':                pop_male,
                        'mo_max_male':             mortality_max_male,
                        'mo_min_male':             mortality_min_male,
                        'mo_mean_male':            mortality_mean_male,
                        'cross_patient_max_male':  cross_patient_max_male,
                        'cross_patient_min_male':  cross_patient_min_male,
                        'cross_patient_mean_male': cross_patient_mean_male,
                        'pop_female':                pop_female,
                        'mo_max_female':             mortality_max_female,
                        'mo_min_female':             mortality_min_female,
                        'mo_mean_female':            mortality_mean_female,
                        'cross_patient_max_female':  cross_patient_max_female,
                        'cross_patient_min_female':  cross_patient_min_female,
                        'cross_patient_mean_female': cross_patient_mean_female,
                        'pop':                   pop_male + pop_female,
                        'mo_total':              mo_total,
                        'mo_total_min':          mo_total_min,
                        'mo_total_max':          mo_total_max,
                        'cross_patient_total':     cross_patient_total,
                        'cross_patient_total_min': cross_patient_total_min,
                        'cross_patient_total_max': cross_patient_total_max,
                    })
        df = pd.DataFrame(mortality)
        for col in ['city', 'disease']:
             df[col] = df[col].astype('category')
        float_cols = df.select_dtypes(include='float64').columns
        df[float_cols] = df[float_cols].astype('float32')
        int_cols = df.select_dtypes(include='int64').columns
        df[int_cols] = df[int_cols].astype('int32')
        df.to_csv(outfile, index=False)
        print(f"[INFO] Saved {outfile}")