#pip install pyomo
import ipdb
import os
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import yaml
#
# Import specific functions from local modules
from Auxiliary_Functions.Process_Input.read_config import process_input_data
from Auxiliary_Functions.Process_Input.read_optional_parameter import process_optional_parameters
from Auxiliary_Functions.Process_Input.date_n_time import date_n_time
from Auxiliary_Functions.Process_Input.read_price_and_generation_scenarios import get_scenarios
from Auxiliary_Functions.Process_Input.process_input_scenarios import read_input_scenarios
from Auxiliary_Functions.Bid_Computation.bid_curve import get_bidcurve
from Auxiliary_Functions.Bid_Formatting.high_low_price_heuristc import get_high_low_price_heuristic
from Auxiliary_Functions.Bid_Formatting.format_bids_pqt import format_bids_pqt_to_PSO
from Auxiliary_Functions.Bid_Formatting.plot_bids import plot_bid_curve
# Define paths for data input and output
datapath= "C:/Anamika/testing/Hybrid-Bidding-test/Hyb_bidding/Input data/"
outpath="C:/Anamika/testing/Hybrid-Bidding-test/Hyb_bidding/Output/"
# Process the input data and load initial configurations
Hybrid_name,POI,esr,gen, ndate=process_input_data(datapath)
optional_data=process_optional_parameters(datapath)
p_thresh_heur=get_scenarios(datapath,ndate,optional_data["p_thresh_heur_pcentile"]) # Process scenarios for stochastic optimization

# Loop through each day to generate bidding strategies
for day in range(0,ndate):
    # Read scenario-specific input data for the given day
    priceDA, gen_data, probS, priceRTadd,Price_Steps,Price_Steps_RT=read_input_scenarios(datapath,day,optional_data)
    # Generate bid curve based on the input data and scenarios
    bid, opt_sol,pqt_bids,pqt_bids_RT = get_bidcurve(esr, gen, POI, optional_data, priceDA, 0.01+priceRTadd, gen_data, probS,Price_Steps,Price_Steps_RT)
    # If price heuristic is enabled and specific bid type is selected, apply high-low price heuristic
    if (optional_data["price_heuristic"]=="yes" and optional_data["bid_type"]==2 ):
       pqt_bids_full=get_high_low_price_heuristic(pqt_bids,opt_sol,priceDA, gen_data, probS,np.arange(24),optional_data,esr,POI,p_thresh_heur)
    else:
        pqt_bids_full=pqt_bids
# Format the bids for submission to a PSO
bid_table_day_ahead =format_bids_pqt_to_PSO(pqt_bids, ndate, esr["max"],Hybrid_name,  24, optional_data)
# Save the DataFrame to a CSV file
bid_table_real_time=format_bids_pqt_to_PSO(pqt_bids_RT, ndate, esr["max"],Hybrid_name,  24, optional_data)
bidoutpath_DA=outpath+"bid_table_day_ahead.csv"
bid_table_day_ahead.to_csv(bidoutpath_DA, index=False)
bidoutpath_RT=outpath+"bid_table_real_time.csv"
bid_table_real_time.to_csv(bidoutpath_RT, index=False)
hour =0
plot_bid_curve(bid_table_day_ahead, outpath, day, hour, optional_data["bid_type"])

