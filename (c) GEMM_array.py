
# --Global Exposure Mortality Model (GEMM) for PM2.5 health impact assessment--#
# author: Shiyu Deng
# affiliation: University College London
# email: shiyu.deng.23@ucl.ac.uk
# date: January 2026
#---refer to www.pnas.org/cgi/doi/10.1073/pnas.1803222115 ---#

import numpy as np
import pandas as pd
import os


def calculate_z(PM2_5):
    z = max(0, PM2_5 - 2.4)
    return z

def gamma_function(PM2_5, disease):
    if disease == 'Ischaemic Heart Disease':
        alpha_hat, mu_hat, pi_hat = 1.9, 12.0, 40.2
    elif disease == 'Strokes':
        alpha_hat, mu_hat, pi_hat = 6.2, 16.7, 23.7
    elif disease == 'Chronic Obstructive Pulmonary Disease':
        alpha_hat, mu_hat, pi_hat = 6.5, 2.5, 32.0
    elif disease == 'Lung Cancer':
        alpha_hat, mu_hat, pi_hat = 6.2, 9.3, 29.8
    elif disease == 'Lower Respiratory Infections':
        alpha_hat, mu_hat, pi_hat = 6.4, 5.7, 8.4
    else:
        raise ValueError("Disease not recognized")
    z = calculate_z(PM2_5)    
    numerator = np.log(1 + z / alpha_hat)
    denominator = 1 + np.exp((mu_hat - z) / pi_hat)
    gamma_value = numerator / denominator
    return gamma_value

def calculate_hazard_ratio(PM2_5, age, disease):   
    z = calculate_z(PM2_5)
    gamma_value = gamma_function(PM2_5, disease)
    
    if disease == 'Ischaemic Heart Disease':
        if 25 <= age < 30:
            theta_hat, se = 0.5070, 0.02458
        elif 30 <= age < 35:
            theta_hat, se = 0.4762, 0.02309
        elif 35 <= age < 40:
            theta_hat, se = 0.4455, 0.0216
        elif 40 <= age < 45:
            theta_hat, se = 0.4148, 0.02011
        elif 45 <= age < 50:
            theta_hat, se = 0.3841, 0.01862
        elif 50 <= age < 55:
            theta_hat, se = 0.3533, 0.01713
        elif 55 <= age < 60:
            theta_hat, se = 0.3226, 0.01564
        elif 60 <= age < 65:
            theta_hat, se = 0.2919, 0.01415
        elif 65 <= age < 70:
            theta_hat, se = 0.2612, 0.01266
        elif 70 <= age < 75:
            theta_hat, se = 0.2304, 0.01117
        elif 75 <= age < 80:
            theta_hat, se = 0.1997, 0.00968
        elif age >= 80:
            theta_hat, se = 0.1536, 0.00745
        else:
            raise ValueError("Age range not recognized for the Ischaemic Heart Disease")

    if disease == 'Strokes':
        if 25 <= age < 30:
            theta_hat, se = 0.4513, 0.11919
        elif 30 <= age < 35:
            theta_hat, se = 0.424, 0.11197
        elif 35 <= age < 40:
            theta_hat, se = 0.3966, 0.10475
        elif 40 <= age < 45:
            theta_hat, se = 0.3693, 0.09752
        elif 45 <= age < 50:
            theta_hat, se = 0.3419, 0.0903
        elif 50 <= age < 55:
            theta_hat, se = 0.3146, 0.08307
        elif 55 <= age < 60:
            theta_hat, se = 0.2872, 0.07585
        elif 60 <= age < 65:
            theta_hat, se = 0.2598, 0.06863
        elif 65 <= age < 70:
            theta_hat, se = 0.2325, 0.06190
        elif 70 <= age < 75:
            theta_hat, se = 0.2051, 0.05418
        elif 75 <= age < 80:
            theta_hat, se = 0.1778, 0.04695
        elif age >= 80:
            theta_hat, se = 0.1368, 0.03611      
        else:
            raise ValueError("Age range not recognized for the Strokese")
    if disease == 'Chronic Obstructive Pulmonary Disease':
        if age >= 25:
            theta_hat, se = 0.2510, 0.06762
        else:
            raise ValueError("Age range not recognized for the Chronic Obstructive Pulmonary Disease")
    if disease == 'Lung Cancer':
        if age >= 25:
            theta_hat, se = 0.2942, 0.06147      
        else:
            raise ValueError("Age range not recognized for the Lung Cancer")
    if disease == 'Lower Respiratory Infections':
        if age >= 25:
            theta_hat, se = 0.4468, 0.11735
        else:
            raise ValueError("Age range not recognized for the Lower Respiratory Infections")

    maxus = (theta_hat + 1.96 * se) * gamma_value
    minus = (theta_hat - 1.96 * se) * gamma_value
    mean = theta_hat * gamma_value
    hazard_ratio_maxus = np.exp(maxus)
    hazard_ratio_minus = np.exp(minus)
    hazard_ratio_mean = np.exp(mean)
    return hazard_ratio_maxus, hazard_ratio_minus, hazard_ratio_mean

def set_base_mortality(age, sex, disease):
    if disease == 'Ischaemic Heart Disease':
        if 25 <= age <= 29:
            if sex == 'male':
                base_mortality = 0.0050*0.01
            elif sex == 'female':
                base_mortality = 0.0027*0.01
        elif 30 <= age < 35:
            if sex == 'male':
                base_mortality = 0.0093*0.01
            elif sex == 'female':
                base_mortality = 0.0045*0.01
        elif 35 <= age < 40:
            if sex == 'male':
                base_mortality = 0.0176*0.01
            elif sex == 'female':
                base_mortality = 0.0081*0.01       
        elif 40 <= age < 45:
            if sex == 'male':
                base_mortality = 0.0318*0.01
            elif sex == 'female':
                base_mortality = 0.0150*0.01         
        elif 45 <= age < 50:
            if sex == 'male':
                base_mortality = 0.0445*0.01
            elif sex == 'female':
                base_mortality = 0.0226*0.01         
        elif 50 <= age < 55:
            if sex == 'male':
                base_mortality = 0.0733*0.01
            elif sex == 'female':
                base_mortality = 0.0415*0.01   
        elif 55 <= age < 60:
            if sex == 'male':
                base_mortality = 0.1154*0.01
            elif sex == 'female':
                base_mortality = 0.0710*0.01            
        elif 60 <= age < 65:
            if sex == 'male':
                base_mortality = 0.1769*0.01
            elif sex == 'female':
                base_mortality = 0.1227*0.01   
        elif 65 <= age < 70:
            if sex == 'male':
                base_mortality = 0.2933*0.01
            elif sex == 'female':
                base_mortality = 0.2174*0.01  
        elif 70 <= age < 75:
            if sex == 'male':
                base_mortality = 0.5145*0.01
            elif sex == 'female':
                base_mortality = 0.4044*0.01            
        elif 75 <= age < 80:
            if sex == 'male':
                base_mortality = 0.8954*0.01
            elif sex == 'female':
                base_mortality = 0.7344*0.01   
        elif age >= 80:
            if sex == 'male':
                base_mortality = 2.7724*0.01
            elif sex == 'female':
                base_mortality = 2.1418*0.01   
        else:
            # Handle unexpected age input
            raise ValueError("Age range not recognized for the Ischaemic Heart Disease")
    elif disease == 'Strokes':
        if 25 <= age <= 29:
            if sex == 'male':
                base_mortality = 0.0041*0.01
            elif sex == 'female':
                base_mortality = 0.0014*0.01
        elif 30 <= age < 35:
            if sex == 'male':
                base_mortality = 0.0082*0.01
            elif sex == 'female':
                base_mortality = 0.0025*0.01
        elif 35 <= age < 40:
            if sex == 'male':
                base_mortality = 0.0156*0.01
            elif sex == 'female':
                base_mortality = 0.0050*0.01       
        elif 40 <= age < 45:
            if sex == 'male':
                base_mortality = 0.0322*0.01
            elif sex == 'female':
                base_mortality = 0.0115*0.01         
        elif 45 <= age < 50:
            if sex == 'male':
                base_mortality = 0.0477*0.01
            elif sex == 'female':
                base_mortality = 0.0203*0.01         
        elif 50 <= age < 55:
            if sex == 'male':
                base_mortality = 0.0878*0.01
            elif sex == 'female':
                base_mortality = 0.0433*0.01   
        elif 55 <= age < 60:
            if sex == 'male':
                base_mortality = 0.1441*0.01
            elif sex == 'female':
                base_mortality = 0.0761*0.01            
        elif 60 <= age < 65:
            if sex == 'male':
                base_mortality = 0.2601*0.01
            elif sex == 'female':
                base_mortality = 0.1421*0.01   
        elif 65 <= age < 70:
            if sex == 'male':
                base_mortality = 0.4586*0.01
            elif sex == 'female':
                base_mortality = 0.2730*0.01  
        elif 70 <= age < 75:
            if sex == 'male':
                base_mortality = 0.9122*0.01
            elif sex == 'female':
                base_mortality = 0.5599*0.01            
        elif 75 <= age < 80:
            if sex == 'male':
                base_mortality = 1.6103*0.01
            elif sex == 'female':
                base_mortality = 1.0058*0.01   
        elif age >= 80:
            if sex == 'male':
                base_mortality = 4.2612*0.01
            elif sex == 'female':
                base_mortality = 2.5671*0.01                                  
        else:
            # Handle unexpected age input
            raise ValueError("Age range not recognized for the Strokes disease")
        
    elif disease == 'Chronic Obstructive Pulmonary Disease':
        if 25 <= age <= 29:
            if sex == 'male':
                base_mortality = 0.0004*0.01
            elif sex == 'female':
                base_mortality = 0.0003*0.01
        elif 30 <= age < 35:
            if sex == 'male':
                base_mortality = 0.0009*0.01
            elif sex == 'female':
                base_mortality = 0.0005*0.01
        elif 35 <= age < 40:
            if sex == 'male':
                base_mortality = 0.0015*0.01
            elif sex == 'female':
                base_mortality = 0.0009*0.01      
        elif 40 <= age < 45:
            if sex == 'male':
                base_mortality = 0.0036*0.01
            elif sex == 'female':
                base_mortality = 0.0018*0.01           
        elif 45 <= age < 50:
            if sex == 'male':
                base_mortality = 0.0063*0.01
            elif sex == 'female':
                base_mortality = 0.0033*0.01       
        elif 50 <= age < 55:
            if sex == 'male':
                base_mortality = 0.0154*0.01
            elif sex == 'female':
                base_mortality = 0.0075*0.01   
        elif 55 <= age < 60:
            if sex == 'male':
                base_mortality = 0.0313*0.01
            elif sex == 'female':
                base_mortality = 0.0156*0.01            
        elif 60 <= age < 65:
            if sex == 'male':
                base_mortality = 0.0724*0.01
            elif sex == 'female':
                base_mortality = 0.0369*0.01   
        elif 65 <= age < 70:
            if sex == 'male':
                base_mortality = 0.1574*0.01
            elif sex == 'female':
                base_mortality = 0.0846*0.01 
        elif 70 <= age < 75:
            if sex == 'male':
                base_mortality = 0.4154*0.01
            elif sex == 'female':
                base_mortality = 0.2256*0.01            
        elif 75 <= age < 80:            
            if sex == 'male':
                base_mortality = 0.8651*0.01
            elif sex == 'female':
                base_mortality = 0.4880*0.01  
        elif age >= 80:
            if sex == 'male':
                base_mortality = 2.9811*0.01
            elif sex == 'female':
                base_mortality = 1.6045*0.01 
        else:
            # Handle unexpected age input
            raise ValueError("Age range not recognized for the Chronic Obstructive Pulmonary Disease")
            
    elif disease == 'Lung Cancer':
        if 25 <= age <= 29:
            if sex == 'male':
                base_mortality = 0.0004*0.01
            elif sex == 'female':
                base_mortality = 0.0003*0.01
        elif 30 <= age < 35:
            if sex == 'male':
                base_mortality = 0.0010*0.01
            elif sex == 'female':
                base_mortality = 0.0006*0.01
        elif 35 <= age < 40:
            if sex == 'male':
                base_mortality = 0.0023*0.01
            elif sex == 'female':
                base_mortality = 0.0014*0.01       
        elif 40 <= age < 45:
            if sex == 'male':
                base_mortality = 0.0054*0.01
            elif sex == 'female':
                base_mortality = 0.0032*0.01         
        elif 45 <= age < 50:
            if sex == 'male':
                base_mortality = 0.0104*0.01
            elif sex == 'female':
                base_mortality = 0.0049*0.01         
        elif 50 <= age < 55:
            if sex == 'male':
                base_mortality = 0.0215*0.01
            elif sex == 'female':
                base_mortality = 0.0093*0.01   
        elif 55 <= age < 60:
            if sex == 'male':
                base_mortality = 0.0396*0.01
            elif sex == 'female':
                base_mortality = 0.0153*0.01            
        elif 60 <= age < 65:
            if sex == 'male':
                base_mortality = 0.0583*0.01
            elif sex == 'female':
                base_mortality = 0.0229*0.01   
        elif 65 <= age < 70:
            if sex == 'male':
                base_mortality = 0.0875*0.01
            elif sex == 'female':
                base_mortality = 0.0340*0.01 
        elif 70 <= age < 75:
            if sex == 'male':
                base_mortality = 0.1279*0.01
            elif sex == 'female':
                base_mortality = 0.0496*0.01            
        elif 75 <= age < 80:            
            if sex == 'male':
                base_mortality = 0.1616*0.01
            elif sex == 'female':
                base_mortality = 0.0643*0.01
        elif age >= 80:
            if sex == 'male':
                base_mortality = 0.2103*0.01
            elif sex == 'female':
                base_mortality = 0.0806*0.01
        else:
            # Handle unexpected age input
            raise ValueError("Age range not recognized for the Lung Cancer Disease")
            
    elif disease == 'Lower Respiratory Infections':
        if 25 <= age <= 29:
            if sex == 'male':
                base_mortality = 0.0008*0.01
            elif sex == 'female':
                base_mortality = 0.0004*0.01
        elif 30 <= age < 35:
            if sex == 'male':
                base_mortality = 0.0010*0.01
            elif sex == 'female':
                base_mortality = 0.0004*0.01
        elif 35 <= age < 40:
            if sex == 'male':
                base_mortality = 0.0013*0.01
            elif sex == 'female':
                base_mortality = 0.0006*0.01       
        elif 40 <= age < 45:
            if sex == 'male':
                base_mortality = 0.0018*0.01
            elif sex == 'female':
                base_mortality = 0.0008*0.01         
        elif 45 <= age < 50:
            if sex == 'male':
                base_mortality = 0.0022*0.01
            elif sex == 'female':
                base_mortality = 0.0009*0.01        
        elif 50 <= age < 55:
            if sex == 'male':
                base_mortality = 0.0035*0.01
            elif sex == 'female':
                base_mortality = 0.0015*0.01  
        elif 55 <= age < 60:
            if sex == 'male':
                base_mortality = 0.0053*0.01
            elif sex == 'female':
                base_mortality = 0.0025*0.01            
        elif 60 <= age < 65:
            if sex == 'male':
                base_mortality = 0.0092*0.01
            elif sex == 'female':
                base_mortality = 0.0046*0.01   
        elif 65 <= age < 70:
            if sex == 'male':
                base_mortality = 0.0160*0.01
            elif sex == 'female':
                base_mortality = 0.0091*0.01  
        elif 70 <= age < 75:
            if sex == 'male':
                base_mortality = 0.0397*0.01
            elif sex == 'female':
                base_mortality = 0.0227*0.01            
        elif 75 <= age < 80:            
            if sex == 'male':
                base_mortality = 0.0884*0.01
            elif sex == 'female':
                base_mortality = 0.0519*0.01   
        elif age >= 80:
            if sex == 'male':
                base_mortality = 0.4738*0.01
            elif sex == 'female':
                base_mortality = 0.2693*0.01 
        else:
            # Handle unexpected age input
            raise ValueError("Age range not recognized for Lower Respiratory Infections")            
    return base_mortality

def PM_mortality(PM2_5, disease, age, sex, population):
    hazard_ratio_max, hazard_ratio_min,hazard_ratio_mean = calculate_hazard_ratio(PM2_5, age, disease)
    base_mortality = set_base_mortality(age, sex, disease)
    mortality_max = (1 - 1 / hazard_ratio_max) * base_mortality * population
    mortality_mean = (1 - 1 / hazard_ratio_mean) * base_mortality * population
    mortality_min = (1 - 1 / hazard_ratio_min) * base_mortality * population
    return mortality_max, mortality_min, mortality_mean
    
def hospital_cost(disease, mortality_max,mortality_min,mortality_mean,age):  
    ##China's legal retirement age is 60 years old and female 55 years old in most cases
    if disease == 'Ischaemic Heart Disease':
           if 25 <= age <= 39:
               hospitalcost_max=mortality_max*3544.9
               hospitalcost_mean=mortality_mean*3544.9
               hospitalcost_min=mortality_min*3544.9
           elif 40 <= age <= 59:
               hospitalcost_max=mortality_max*3544.9
               hospitalcost_mean=mortality_mean*3544.9
               hospitalcost_min=mortality_min*3544.9
           elif 60 <= age <= 79:
               hospitalcost_max=mortality_max*4323.3
               hospitalcost_mean=mortality_mean*4323.3
               hospitalcost_min=mortality_min*4323.3
           elif 80 <= age :
               hospitalcost_max=mortality_max*5491.0 
               hospitalcost_mean=mortality_mean*5491.0
               hospitalcost_min=mortality_min*5491.0

    elif disease == 'Strokes':
           if 25 <= age <= 39:
               hospitalcost_max=mortality_max*1661.9
               hospitalcost_mean=mortality_mean*1661.9
               hospitalcost_min=mortality_min*1661.9
           elif 40 <= age <= 59:
               hospitalcost_max=mortality_max*1824.4 
               hospitalcost_mean=mortality_mean*1824.4
               hospitalcost_min=mortality_min*1824.4 
           elif 60 <= age <= 79:
               hospitalcost_max=mortality_max*1771.9
               hospitalcost_mean=mortality_mean*1771.9
               hospitalcost_min=mortality_min*1771.9
           elif 80 <= age :
               hospitalcost_max=mortality_max*2109.0 
               hospitalcost_mean=mortality_mean*2109.0
               hospitalcost_min=mortality_min*2109.0

    elif disease == 'Chronic Obstructive Pulmonary Disease':
           if 25 <= age <= 39:
               hospitalcost_max=mortality_max*3611.9
               hospitalcost_mean=mortality_mean*3611.9
               hospitalcost_min=mortality_min*3611.9
           elif 40 <= age <= 59:
               hospitalcost_max=mortality_max*3219.3
               hospitalcost_mean=mortality_mean*3219.3
               hospitalcost_min=mortality_min*3219.3
           elif 60 <= age <= 79:
               hospitalcost_max=mortality_max*4318.5
               hospitalcost_mean=mortality_mean*4318.5
               hospitalcost_min=mortality_min*4318.5
           elif 80 <= age :
               hospitalcost_max=mortality_max*5731.9
               hospitalcost_mean=mortality_mean*5731.9
               hospitalcost_min=mortality_min*5731.9
              
    elif disease == 'Lung Cancer':
           if 25 <= age <= 39:
               hospitalcost_max=mortality_max*5566.5
               hospitalcost_mean=mortality_mean*5566.5
               hospitalcost_min=mortality_min*5566.5
           elif 40 <= age <= 59:
               hospitalcost_max=mortality_max*6071.9 
               hospitalcost_mean=mortality_mean*6071.9
               hospitalcost_min=mortality_min*6071.9 
           elif 60 <= age <= 79:
               hospitalcost_max=mortality_max*5848.7
               hospitalcost_mean=mortality_mean*5848.7
               hospitalcost_min=mortality_min*5848.7
           elif 80 <= age :
               hospitalcost_max=mortality_max*5140.7
               hospitalcost_mean=mortality_mean*5140.7
               hospitalcost_min=mortality_min*5140.7
              
    elif disease == 'Lower Respiratory Infections':
           if 25 <= age <= 39:
               hospitalcost_max=mortality_max*1267.3
               hospitalcost_mean=mortality_mean*1267.3
               hospitalcost_min=mortality_min*1267.3
           elif 40 <= age <= 59:
               hospitalcost_max=mortality_max*1267.3
               hospitalcost_mean=mortality_mean*1267.3
               hospitalcost_min=mortality_min*1267.3
           elif 60 <= age <= 79:
               hospitalcost_max=mortality_max*2100.8
               hospitalcost_mean=mortality_mean*2100.8
               hospitalcost_min=mortality_min*2100.8  
           elif 80 <= age :
               hospitalcost_max=mortality_max*2100.8
               hospitalcost_mean=mortality_mean*2100.8
               hospitalcost_min=mortality_min*2100.8
   
    else:
        hospitalcost_max=0
        hospitalcost_mean=0
        hospitalcost_min=0       
        print(f"Warning: Disease {disease} is not handled for female hospital cost calculation.")
    
    return hospitalcost_max, hospitalcost_min,hospitalcost_mean
            
def cross_region(mortality_max,mortality_min,mortality_mean,age,mobility): 
     if 25 <= age <= 29:
       if mobility*1.1323*1.05<=1:  
         cross_patient_max=mortality_max*mobility*1.1323*1.05
       else:
         cross_patient_max=mortality_max*1
       cross_patient_mean=mortality_mean*mobility*1.1323
       cross_patient_min=mortality_min*mobility*1.1323*0.95
     elif 30 <= age <= 39:
       if mobility*0.8169*1.05<=1:
         cross_patient_max=mortality_max*mobility*0.8160*1.05
       else:
         cross_patient_max=mortality_max*1
       cross_patient_mean=mortality_mean*mobility*0.8160
       cross_patient_min=mortality_min*mobility*0.8160*0.95
     elif 40 <= age <= 49:
       if mobility*0.9465*1.05<=1:
         cross_patient_max=mortality_max*mobility*0.9465*1.05
       else:
         cross_patient_max=mortality_max*1
       cross_patient_mean=mortality_mean*mobility*0.9465
       cross_patient_min=mortality_min*mobility*0.9465*0.95
     elif 50 <= age <= 59:
       if mobility*1.1176*1.05<=1:
         cross_patient_max=mortality_max*mobility*1.1176*1.05
       else:
         cross_patient_max=mortality_max*1
       cross_patient_mean=mortality_mean*mobility*1.1176
       cross_patient_min=mortality_min*mobility*1.1176*0.95
     elif 60 <= age :
       if mobility*0.9874*1.05<=1:
         cross_patient_max=mortality_max*mobility*0.9874*1.05
       else:
         cross_patient_max=mortality_max*1
       cross_patient_mean=mortality_mean*mobility*0.9874
       cross_patient_min=mortality_min*mobility*0.9874*0.95      
     return cross_patient_max, cross_patient_min,cross_patient_mean
 
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
               
