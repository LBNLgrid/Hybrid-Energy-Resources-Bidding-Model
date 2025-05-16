# Hybrid Energy Resources Bidding Model (HERB)

### Overview
- Hybrid Energy Resources Bidding Model (HERB) is a tool designed to generate hourly bid curves for a day-ahead market, catering to a single, self-managed resource participation hybrid model.
- This tool addresses the uncertainty in renewable generation and market prices.
- Users can generate regular, self-schedule, and stair-step bid curves according to their requirements.

The tool frames the problem as either a Mixed Integer Linear Programming (MILP) or Linear Programming (LP) optimization, based on user preference.

### Prerequisites

- **Python Version:** Python 3.6 or higher.
- **Required Libraries:** All required Python libraries are listed in the `requirements.txt` file.
- **Solver:** A MILP or LP solver must be installed. CBC is recommended for its free availability.

### Installing

1. **Library Installation:**
   Install the necessary Python libraries using the provided requirements file:
   ```
   pip install -r requirements.txt
   ```

To install the package, use

```
pip setup.py install
```
CBC can be installed for all platforms. For installation instructions, check the [github repository](https://github.com/coin-or/Cbc). The solver can be changed by modifying Others/solver in test/test_runs/parameters.xlsx.

To switch the solver to CBC in your Pyomo models, add the following line to your code:

```
solver = pyo.SolverFactory('cbc')
```

## Running

You can run the model by executing the following command in the main folder:

```
Hyb_bidding/main.py
```
## Code Structure 
![image](https://github.com/AnaTiw/Hybrid-Bidding/assets/157315954/53de3401-3023-45a2-a336-4b682692782e)

## Input data
Input data samples  are available in the Hyb_bidding/Input data folder.
The optimization problem parameters can be adjusted in the config.yml and opt_param. yml  located in the Hyb_bidding/Input data.

## Output
This tool outputs two types of bid tables: bid_table_day_ahead.csv and bid_table_real_time.csv. These tables contain structured bidding data pertinent to energy markets. Below is a detailed breakdown of each column present in these bid tables:

-Price_USD_per_MWh: This column lists the bid price in U.S. dollars per megawatt-hour. 

-Quantity_MW: This field specifies the bid quantity in Megawatt. 

-Hour: Denotes the hour of the day for which the bid is valid, following a 24-hour format from 0 (midnight) to 23 (11 PM).

-Day: Indicates the day of the month, the bid corresponds to. 

-Point:  This column specifies the index of the bid point for each row, indicating the position in the bidding sequence. This can be negative if the bid is for charging the hybrid, or positive if the bid is for discharging.

-maxSOC_MWh: Defined by the user in a “yaml” file, this column shows the maximum State of Charge in megawatt-hours.

-Hybrid_Name:  Lists the name of the hybrid system as specified by the user in the  “yaml” file.
  
## Copyright Notice
*** Copyright Notice ***

Hybrid Energy Resources Bidding Model (HERB) Copyright (c) 2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.