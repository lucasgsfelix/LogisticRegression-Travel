## Extract cities home towns lat, lon

import pandas as pd


if __name__ == '__main__':


	pandas_df = pd.read_csv("yelp_enriched_dataset.csv", sep=';')


	pandas_df['user_visited_city'] = pandas_df['city'] + '_' + pandas_df['state'] + pandas_df['country_name']


	columns = ['user_visited_city', 'latitude', 'longitude']

	pandas_df = pandas_df[columns].groupby("user_visited_city").mean().reset_index()


	pandas_df.to_csv("geolocation_user_hometown.csv", sep=';', index=False)
