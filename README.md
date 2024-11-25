# Hybrid-Bidding

### Overview
-Hybrid-Bidding is a tool designed to generate hourly bid curves for a day-ahead market, catering to a single, self-managed resource participation hybrid model.
-This tool addresses the uncertainty in renewable generation and market prices.
-Users can generate regular, self-schedule, and stair-step bid curves according to their requirements.

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

## Running

You can run the model by executing the following command in the main folder:

```
Hyb_bidding/main.py
```
## Test Systems Dataset

Input data samples  are available in the Hyb_bidding/Input data folder, providing insight of the capabilities of the tool.
The optimization problem parameters can be adjusted in the config.yml and opt_param. yml  located in the Hyb_bidding/Input data.


  
