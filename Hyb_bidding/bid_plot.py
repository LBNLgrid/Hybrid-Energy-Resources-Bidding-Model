
import pandas as pd
import os
from Auxiliary_Functions.Process_Input.read_optional_parameter import process_optional_parameters
from Auxiliary_Functions.Bid_Formatting.plot_bids import plot_bid_curve
 # Initialize variables
hour =0
day=0
datapath= "C:/Anamika/testing/Hybrid-Bidding-test/Hyb_bidding/Input data/"
outpath="C:/Anamika/testing/Hybrid-Bidding-test/Hyb_bidding/Output/"
# Define file paths
bidoutpath_DA = os.path.join(outpath, "bid_table_day_ahead.csv")
bidoutpath_RT = os.path.join(outpath, "bid_table_real_time.csv")
optional_data=process_optional_parameters(datapath)
if not os.path.exists(bidoutpath_DA):
    raise FileNotFoundError("File is missing: " + bidoutpath_DA)
if not os.path.exists(bidoutpath_RT):
    raise FileNotFoundError("File is missing: " + bidoutpath_RT)
bid_table_day_ahead=pd.read_csv(bidoutpath_DA)
bid_table_real_time=pd.read_csv(bidoutpath_RT)
plot_bid_curve(bid_table_day_ahead, outpath, day, hour, optional_data["bid_type"])