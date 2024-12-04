

#from get_price_scenario import stair_breaks_from_scen


import numpy as np
import pandas as pd
def adjust_column_names(df, prefix):
    df.columns = [f"{prefix}_{col}" for col in df.columns]
    return df
def get_scenarios(datapath,ndate,p_thresh_heur_pcentile):
    """
        Aggregates and preprocesses daily energy market data from multiple sources into a single dataset.

        This function processes input data for a specified number of days from given directories. It reads separate Excel files for price data, generator data, and probability scenarios for each day, standardizes column names with specified prefixes, and merges them into a single DataFrame. The function finally concatenates data from all days and saves it as a Parquet file for efficient storage and access.

        Parameters:
            datapath (str): The base directory path where the data files are stored. It assumes a specific sub-directory structure and file naming convention.
            ndate (int): The number of days for which the data is to be processed, starting from zero.

        Returns:
            None: The function does not return any value but saves the processed data to a Parquet file in the specified directory path.
    """

    dir_current = datapath + "Price_and_Generation_Scenario/"
    combined_all_days = pd.DataFrame()
    provided_scenarios=pd.read_excel(dir_current + "scenarios_per_day.xlsx", engine='openpyxl')
    for day in range (0,ndate) :
        total_scenarios = provided_scenarios['Total_Number_of_Scenarios'].iloc[day]
    #temp=pd.ExcelFile(dir_current+"pricedata3.xlsx")
        pDA = pd.read_excel(dir_current+"price_data_day_ahead.xlsx",sheet_name=day, engine='openpyxl')
        if (len(pDA) == total_scenarios):
            if 'Scenario' in pDA.columns:
                pDA =pDA.drop(columns=['Scenario'])
        else:
             raise ValueError("Missing Scenario in price_data_day_ahead ")
        genDA = pd.read_excel(dir_current+"available_generation_data.xlsx",sheet_name=day, engine='openpyxl')
        if (len(genDA) == total_scenarios):
           if 'Scenario' in genDA.columns:
               genDA = genDA.drop(columns=['Scenario'])
        else:
             raise ValueError("Missing Scenario in available_generation_data ")
        probSc = pd.read_excel(dir_current+"scenarios_probabilities.xlsx",sheet_name=day, engine='openpyxl')
        if (len(probSc) == total_scenarios):
           if 'Scenario' in probSc.columns:
               probSc = probSc.drop(columns=['Scenario'])
               #check if the sum of probabilities is exactly 1
               prob_sum=probSc.sum()
               if prob_sum.iloc[0].astype(int) ==1 :
                   pass
               else:
                   raise ValueError("Sum of scenario probabilities should be 1 ")
        else:
             raise ValueError("Missing Scenario in scenarios_probabilities ")
        pRT = pd.read_excel(dir_current+"price_data_real_time.xlsx",sheet_name=day, engine='openpyxl')
        if (len(pRT) == total_scenarios):
           if 'Scenario' in pRT.columns:
               pRT = pRT.drop(columns=['Scenario'])
        else:
           raise ValueError("Missing Scenario in price_data_real_time ")

            # Adjusting column names
        price_data = adjust_column_names(pDA, 'pd')
        priceRTadd = adjust_column_names(pRT, 'pRT')
        gen_data = adjust_column_names(genDA, 'gd')
        prob_s = adjust_column_names(probSc, 'prob')
        pDA['pday'] = day
        genDA['gday'] = day
        probSc['psday'] = day
        priceRTadd['pRTday'] = day
        # Combine data into a single DataFrame
        combined_data = pd.concat([price_data, priceRTadd,gen_data, prob_s], axis=1)
        # Save to Parquet file
        combined_all_days = pd.concat([combined_all_days, combined_data],ignore_index=True)
    outputpath=f"{dir_current}parquet_data_combined"
    combined_all_days.to_parquet(outputpath)
    p_low=np.quantile(combined_all_days.iloc[:,0:48],p_thresh_heur_pcentile[0])
    p_up = np.quantile(combined_all_days.iloc[:, 0:48], p_thresh_heur_pcentile[1])
    p_thresh_heur=np.array([p_low,p_up])
    return p_thresh_heur