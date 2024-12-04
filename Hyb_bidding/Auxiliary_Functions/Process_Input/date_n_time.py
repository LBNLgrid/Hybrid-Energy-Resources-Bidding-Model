from datetime import datetime, timedelta
import pytz

# Create timezone object for Eastern Time
def date_n_time (opt_data):
    # 'datetimes' contains all the datetime objects for each hour over 31 days
    year=opt_data["start_date"][0]
    month = opt_data["start_date"][1]
    day = opt_data["start_date"][2]
    tz= pytz.timezone(opt_data["timezone"])

# Start date
    start_date = datetime(year, month, day, 0, 0, tzinfo=tz)

# List to hold all the datetime objects
    datetimes = []
    dates=[]

# Generate datetimes for 31 days, each day having 24 hours
    for day in range(31):
       current_day = start_date + timedelta(days=day)
       dates.append(current_day)
       for hour in range(24):
           datetimes.append(current_day + timedelta(hours=hour))

# Now 'datetimes' contains all the datetime objects for each hour over 31 days
    return dates,datetimes
