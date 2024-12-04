import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

def plot_bid_curve(bid_data, output_folder, day, hour, bid_type):
    """
    Plots the bid curve for a specified day and hour, for either Day Ahead or Real Time market.

    Parameters:
        bid_data (str): Table containing bid information.
        output_folder (str): Path to the directory where plots will be saved.
        day (int): The day for which the bid curve is required.
        hour (int): The hour for which the bid curve is required.
        bid_type (str): 'DA' for Day Ahead or 'RT' for Real Time.
    """
    if(bid_type==1):
       market_type='Self_Schedule'
    elif(bid_type==2):
        market_type = 'Stair_Step'
    else:
        market_type = 'Regular'
    # Filter the data for the specified day and hour
    bid_curve = bid_data[(bid_data['Day'].astype(np.float64).astype(np.int32)  == day) & (bid_data['Hour'].astype(np.float64).astype(np.int32) == hour)]

    if bid_curve.empty:
        print(f"No data available for day {day}, hour {hour} in {market_type} market.")
        return

    # Plotting the bid curve
    plt.figure(figsize=(10, 6))
    plt.step(bid_curve['Price_USD_per_MWh'], bid_curve['Quantity_MW'], where='post',marker='o', linestyle='-')
    plt.title(f'Bid Curve for Day {day}, Hour {hour} ({market_type})')
    plt.xlabel('Price (USD_per_MWh)')
    plt.ylabel('Quantity(MW)')
    plt.grid(True)

    # Improve tick formatting to enhance readability
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # Ensure everything is nicely laid out to avoid cropping
    plt.tight_layout()
    # Save the plot
    plot_filename = f"{output_folder}/bid_curve_day_{day}_hour_{hour}_{market_type}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot saved as {plot_filename}")

