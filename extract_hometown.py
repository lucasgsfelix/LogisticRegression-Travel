## Extract cities home towns lat, lon

import pandas as pd


if __name__ == '__main__':


	pandas_df = pd.read_csv("yelp_enriched_dataset.csv", sep=';')

	print("Quantidade de linhas antes do dropnat: ", len(pandas_df))

	pandas_df['date'] = pd.to_datetime(pandas_df['date'], errors='coerce')

	pandas_df = pandas_df.dropna(subset='date')

	print("Quantidade de linhas depois do dropnat: ", len(pandas_df))

	pandas_df['user_visited_city'] = pandas_df['city'] + '_' + pandas_df['state'] + pandas_df['country_name']


	columns = ['user_visited_city', 'latitude', 'longitude']

	pandas_df = pandas_df[columns].groupby("user_visited_city").mean().reset_index()


	pandas_df.to_csv("geolocation_user_hometown.csv", sep=';', index=False)
