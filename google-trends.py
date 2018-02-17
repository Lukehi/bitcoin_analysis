from datetime import datetime, timedelta
from pytrends.request import TrendReq
import pandas as pd
import os
directory = '/Users/lukehindson/PycharmProjects/Bitcoin/Data/Google'
os.chdir(directory)
filename = 'btc_googletrends.csv'

# Max requests is 270 records. Include overlap for scaling
maxstep = 269
overlap = 10
step = maxstep - overlap + 1
kw_list = ['bitcoin','btc']
start_date = datetime(2012, 9, 13).date()

# Login to Google.
pytrend = TrendReq(hl='en-US,tz=360')

# Run first iteration
today = datetime.today().date()
old_date = today

# Go back in time number of steps
new_date = today - timedelta(days=step)

# Hold data in interest_over_time_df
timeframe = new_date.strftime('%Y-%m-%d') + ' ' + old_date.strftime('%Y-%m-%d')
pytrend.build_payload(kw_list=kw_list, timeframe=timeframe)
interest_over_time_df = pytrend.interest_over_time()

# Iterate over time frames

while new_date > start_date:
	### Save the new date from the previous iteration.
	# Overlap == 1 would mean that we start where we
	# stopped on the iteration before, which gives us
	# indeed overlap == 1.
	old_date = new_date + timedelta(days=overlap - 1)

	### Update the new date to take a step into the past
	# Since the timeframe that we can apply for daily data
	# is limited, we use step = maxstep - overlap instead of
	# maxstep.
	new_date = new_date - timedelta(days=step)
	# If we went past our start_date, use it instead
	if new_date < start_date:
		new_date = start_date

	# New timeframe
	timeframe = new_date.strftime('%Y-%m-%d') + ' ' + old_date.strftime('%Y-%m-%d')
	print(timeframe)

	# Download data
	pytrend.build_payload(kw_list=kw_list, timeframe=timeframe)
	temp_df = pytrend.interest_over_time()
	#temp_df.to_csv(timeframe.replace(' ', '_')+'.csv')
	if (temp_df.empty):
		raise ValueError(
			'Google sent back an empty dataframe. Possibly there were no searches at all during the this period! Set start_date to a later date.')
	# Renormalize the dataset and drop last line
	for kw in kw_list:
		beg = new_date
		end = old_date - timedelta(days=1)

		# Since we might encounter zeros, we loop over the
		# overlap until we find a non-zero element
		for t in range(1, overlap + 1):
			# print('t = ',t)
			# print(temp_df[kw].iloc[-t])
			if temp_df[kw].iloc[-t] != 0:
				scaling = float(interest_over_time_df[kw].iloc[t - 1]) / float(temp_df[kw].iloc[-t])
				# print('Found non-zero overlap!')
				break
		# Apply scaling
		print scaling
		temp_df.loc[beg:end, kw] = temp_df.loc[beg:end, kw] * scaling
	#time.sleep(randint(5, 10))

	interest_over_time_df = pd.concat([temp_df[:-overlap], interest_over_time_df])

# Save dataset
interest_over_time_df.to_csv(filename)
