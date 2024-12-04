
import pandas as pd
import numpy as np
from Auxiliary_Functions.Process_Input.get_price_scenario import stair_breaks_from_scen
def load_and_extract_data(file_path):
    try:
        # Load the entire DataFrame from a Parquet file
        data = pd.read_parquet(file_path)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None


def columns_to_numpy(data, prefix, day_column, day):
    if data is not None:
        # Filter data for the specific day
        filtered_data = data[data[day_column] == day]

        # Extract columns by prefix and convert to numpy array
        filtered_columns = filtered_data.filter(regex=f'^{prefix}').columns
        if not filtered_columns.empty:
            return filtered_data[filtered_columns].to_numpy()
        else:
            print(f"No columns found with prefix '{prefix}' for day {day}")
            return None
    else:
        print("Data is not available to extract.")
        return None
def read_input_scenarios(data_path,day,opt_data):
    """
        Reads and processes scenario data from a Parquet file, filtering and transforming the data for a specific day.

        This function loads combined energy market data from a Parquet file and filters it for a specified day. It extracts specific columns related to price, generation, and probability scenarios, converts them to numpy arrays, and applies further processing. The function also calculates price steps for day-ahead market scenarios using a custom function based on the filtered data.

        Parameters:
            data_path (str): The base directory path where the Parquet file is stored.
            day (int): The day for which data is to be filtered and processed.
            opt_data (np.array): An array of options data used in calculating price steps.

        Returns:
            tuple: A tuple containing numpy arrays for day-ahead price data, generator data, probability scenarios, real-time price additions, and price steps.

            - priceDA (np.array): Array of day-ahead prices.
            - gen_data (np.array): Array of generator data.
            - probS (np.array): Array of probability scenarios.
            - priceRTadd (np.array): Array of real-time price additions.
            - PRICE_STEPS (np.array): Calculated price steps for the scenarios.
     """


    dir_current = f"{data_path}Price_and_Generation_Scenario/parquet_data_combined"
    data = pd.read_parquet(dir_current)
    # Filter rows where the day column matches the specified day
    #price_steps_np = columns_to_numpy(data, 'ps_')
    priceDA = columns_to_numpy(data, 'pd_', 'pday', day)
    priceRTadd = columns_to_numpy(data, 'pRT_', 'pRTday', day)
    gen_data = columns_to_numpy(data, 'gd_', 'gday', day)
    probS = columns_to_numpy(data, 'prob_', 'psday', day)
    PRICE_STEPS=stair_breaks_from_scen(priceDA,opt_data,probS*np.ones((1,48)))
    ### generate price steps for real time price data
    if np.any(priceRTadd) :
        PRICE_STEPS_RT = stair_breaks_from_scen(priceRTadd, opt_data, probS * np.ones((1, 48)))
    else :
        PRICE_STEPS_RT = np.array([])

    #priceRTadd=np.zeros((200,48))
    return priceDA, gen_data, probS,priceRTadd,PRICE_STEPS,PRICE_STEPS_RT