'''
Collect daily data from Google Trends and store to CSV
'''

from datetime import datetime, timedelta
from pytrends.request import TrendReq
import pandas as pd
import os


directory = '/Users/lukehindson/PycharmProjects/Bitcoin/Data/Google'
os.chdir(directory)
filename = 'btc_googletrends.csv'

# Max requests is 270 records. Include overlap for scaling purposes
maxstep = 269
overlap = 10
step = maxstep - overlap + 1
kws = ['bitcoin']
start = datetime(2012, 9, 13).date()

# Login to Google.
pytrend = TrendReq(hl='en-US,tz=360')

# Run first iteration
now = datetime.today().date()

# Go back in time number of steps
new_date = today - timedelta(days=step)

# Hold data in interest_over_time_df
timeframe = new_date.strftime('%Y-%m-%d') + ' ' + now.strftime('%Y-%m-%d')
pytrend.build_payload(kws=kws, timeframe=timeframe)
interest_over_time_df = pytrend.interest_over_time()

# Iterate over time frames
while new_date > start:
	# Save the new date from the previous iteration.
	# Overlap == 1 would means that we start where we stopped on the iteration before
	now = new_date + timedelta(days=overlap - 1)

	# Update the new date to take a step into the past
	# The daily timeframe is limited so use step = maxstep - overlap instead of maxstep
	new_date = new_date - timedelta(days=step)
	# If we went past our start, use it instead
	if new_date < start:
		new_date = start

	# New timeframe
	timeframe = new_date.strftime('%Y-%m-%d') + ' ' + now.strftime('%Y-%m-%d')
	print(timeframe)

	# Download data
	pytrend.build_payload(kws=kws, timeframe=timeframe)
	temp_df = pytrend.interest_over_time()

	if (temp_df.empty):
		raise ValueError('Google sent back an empty dataframe.')

	# Renormalize the dataset and drop last line
	for kw in kws:
		beg = new_date
		end = now - timedelta(days=1)

		# Since we might encounter zeros, we loop over the overlap until we find a non-zero element
		for t in range(1, overlap + 1):
			if temp_df[kw].iloc[-t] != 0:
				scaling = float(interest_over_time_df[kw].iloc[t - 1]) / float(temp_df[kw].iloc[-t])
				break

		# Apply scaling
		print scaling
		temp_df.loc[beg:end, kw] = temp_df.loc[beg:end, kw] * scaling

	interest_over_time_df = pd.concat([temp_df[:-overlap], interest_over_time_df])

# Save dataset
interest_over_time_df.to_csv(filename)
