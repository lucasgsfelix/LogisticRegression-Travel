import pandas as pd
import tqdm
import os
import numpy as np
import subprocess

def read_chuncks(chuncks):
    
    for chunck in chuncks:
    
    
        #chunck.to_csv("reviews_chunck.csv", sep=';', index=False)

        break
    
    return chunck


if __name__ == '__main__':


	pandas_df = pd.read_csv("user_trips_table_plus_home_geo.csv", sep=';')
	#pandas_df = pd.read_csv("users_trips.csv", sep=';')

	print("Quantidade de linhas antes do dropnat: ", len(pandas_df))

	pandas_df['date'] = pd.to_datetime(pandas_df['date'], errors='coerce')

	pandas_df = pandas_df.dropna(subset=['date'])

	print("Quantidade de linhas depois do dropnat: ", len(pandas_df))

	batches = np.array_split(pandas_df['user_id'].unique(), 1)

	try:

		os.mkdir("Datasets")


	except:

		pass


	cut_off = 0.00

	cut_offs = [cut_off/100 for cut_off in range(50, 105, 5)]

	for users in tqdm.tqdm(batches):

		pandas_df[pandas_df['user_id'].isin(users)].to_csv("Datasets/yelp_users_batches_" + str(cut_off) + ".csv", sep=';', index=False)
	
		subprocess.call("python3 generate_travels_table.py " + str(cut_off), shell=True)

