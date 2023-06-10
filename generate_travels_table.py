import tqdm
import pandas as pd
import numpy as np
from geopy import distance
import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool
from functools import partial

import sys
import os



def generate_trip_type_dataframe(df, work_porc):

    df_predictions = pd.DataFrame(predictions, columns=['predictions'])

    df_predictions['review_id'] = index_predictions['review_id']

    df_trip_types = df[['trip_id', 'review_id']].set_index('review_id').join(df_predictions.set_index('review_id')).reset_index()

    qtde_places = df_trip_types['trip_id'].value_counts().reset_index().rename(columns={'index': 'trip_id', 'trip_id': 'qtde_places'})

    qtde_work = df_trip_types.groupby("trip_id").sum().reset_index().rename(columns={'predictions': 'qtde_work'})

    df_trip_types = qtde_places.set_index('trip_id').join(qtde_work.set_index('trip_id')).reset_index()

    df_trip_types['porc'] = df_trip_types['qtde_work']/df_trip_types['qtde_places']

    df_trip_types['trip_type'] = 0

    df_trip_types.loc[df_trip_types['porc'] >= work_porc, 'trip_type'] = 1

    return df_trip_types


def calculate_distance(row, suffix):

    if np.isnan(row['latitude' + suffix]) or np.isnan(row['longitude' + suffix]):
        
        return 0
    
    # lat, lon
    return distance.distance((row['latitude'], row['longitude']),
                             (row['latitude' + suffix], row['longitude' + suffix])).km



def retrieve_features(user):

    df_trip = df[df['trip_id'] == user]

    trip_dict = {}
    
    trip_dict['date'] = df_trip['date'].dt.date.min()

    trip_dict['amount_places'] = len(df_trip)
    
    try:

        trip_dict['travel_duration'] = (df_trip['date'].dt.date.max() - df_trip['date'].dt.date.min()).days + 1
    
    except:

        trip_dict['travel_duration'] = 1
    
    trip_dict['travel_duration'] = int(trip_dict['travel_duration'])
    
    # tempo médio em dias entre um checkin e outro
    trip_dict['check_in_freq'] = len(df_trip)/trip_dict['travel_duration']

    # tempo médio de duração entre um checkin e outro - em horas
    df_trip['date_shift'] = df_trip['date'].shift(-1)
    
    try:
    
        # tempo médio entre os checkins
        trip_dict['check_in_dura'] = np.mean(np.abs((df_trip['date_shift'].dt.date - df_trip['date'].dt.date).dt.total_seconds()/60))
    
    except:
        
        trip_dict['check_in_dura'] = 0

    # quantidade de categorias diferentes visitadas
    trip_dict['categories'] = len(df_trip['categories'].unique())
    
    # mean check-in time
    trip_dict['check_in_time'] = np.mean(df_trip['date'].dt.hour)

    # distancia média entre os locais visitados
    df_trip['latitude_shift'] = df_trip['latitude'].shift(-1)

    df_trip['longitude_shift'] = df_trip['longitude'].shift(-1)

    columns = ['latitude', 'longitude', 'latitude_shift', 'longitude_shift']

    try:
    
        df_trip['distance'] = df_trip[columns].apply(lambda x: calculate_distance(x, '_shift'), axis=1)
    
    except:
        
        df_trip['distance'] = 0

    trip_dict['mean_distance'] = np.mean(df_trip['distance'])

    trip_dict['radius_gyration'] = np.mean(np.abs(df_trip['distance'] - trip_dict['mean_distance']))
    
    trip_dict['user_id'] = user

    trip_dict['trip_id'] = df_trip['trip_id'].values[0]

    # work or leiusre
    # 1 to work, 0 to leisure
    trip_dict['trip_type'] = df_trip['trip_type'].values[0]

    trip_dict['average_start'] = df_trip['stars'].mean()

    df_trip['categories'] = df_trip['categories'].str.lower()


    # categorias visitadas
    for category in ['Restaurant', 'Hotel', 'Automotive', 'Attraction']:
        
        trip_dict[category] = len(df_trip[df_trip['category'] == category])

    trip_month = df_trip['date'].dt.month.min()

    columns = ['latitude', 'longitude', 'latitude_hotel', 'longitude_hotel']

    if trip_dict['Hotel'] > 1:

        df_trip['latitude_hotel'] = df_trip[df_trip['categories'].str.contains('hotel') == True]['latitude'].values[0]

        df_trip['longitude_hotel'] = df_trip[df_trip['categories'].str.contains('hotel') == True]['longitude'].values[0]

        df_trip['hotel_distance'] = df_trip[columns].apply(lambda x: calculate_distance(x, '_hotel'), axis=1)
    
        trip_dict['hotel_distance'] = df_trip['hotel_distance'].mean()
    
    else:
        
        ### nesse caso não tem hotel na história
        trip_dict['hotel_distance'] = df_trip['distance'].max()
    
    
    df_trip['hours'] = df_trip['date'].dt.hour

    df_trip['hours'] = df_trip['hours'].astype(int)
    
    ## Porcentagem de checkins dentro do horário comercial
    trip_dict['perc_business_hours'] = len(df_trip[(df_trip['hours'] >= 9) &
                                                   (df_trip['hours'] <= 18)])/len(df_trip)

    # desvio padrão entre as horas
    trip_dict['check_in_time_dp'] = np.std(df_trip['date'].dt.hour, ddof=1)
    
    # Mẽs da viagem
    for month in range(1, 13):
        
        trip_dict[str(month)] = 0

    trip_dict[str(trip_month)] = 1
    
    return trip_dict


def standard_categories(df):

    df['category'] = 'Attraction'

    for category in ['Restaurant', 'Hotel', 'Food', 'Automotive']:

        df.loc[df['categories'].str.contains(category) == True, 'category'] = category

        
    df.loc[df['category'] == ('Food'), 'category'] = 'Restaurant'

    return df


def evaluate_user_travels(user):

    user_trips_df = df_trips[df_trips['user_id'] == user]
    
    qtde_trips = len(user_trips_df['trip_id'].unique())

    user_trips_df = user_trips_df.sort_values('date')

    new_features_list, new_names = [], {column: column + '_new_travel' for column in user_trips_df.columns}

    sum_columns = [str(value) for value in range(1, 13)] + ['Hotel', 'Attraction', 'Restaurant', 'Automotive']
    
    columns_to_drop = ['trip_type'] + sum_columns

    for date in user_trips_df['date'].values:

        query = user_trips_df['date'] <= date

        # média das viagens feitas no passado
        past_mean_df = user_trips_df[query].drop('date', axis=1).mean().to_frame().T.drop(columns_to_drop, axis=1)

        past_sum_df = user_trips_df[user_trips_df['date'] < date][sum_columns].sum().to_frame().T

        # quantidade de viagens feitas anteriormente, desconsideramos a viagem atual
        past_mean_df['qtde_past_trips'] = len(user_trips_df[query]) - 1

        # quantidade de viagens feitas anteriormente que eram a trabalho
        past_mean_df['past_work_travel'] = user_trips_df[user_trips_df['date'] < date]['trip_type'].sum()

        
        past_mean_df = past_mean_df.join(past_sum_df)
        
        new_features_list.append(past_mean_df)


    past_trips_df = pd.concat(new_features_list).rename(columns=new_names)

    return user_trips_df.reset_index(drop=True).join(past_trips_df).reset_index(drop=True)




work_cutoff = float(sys.argv[1])

# 1 é viagem a trabalho

df = pd.read_table("Datasets/yelp_users_batches_" + str(work_cutoff) + ".csv", sep=';')

index_predictions = pd.read_table("id_predictions.csv", sep=',').drop('Unnamed: 0', axis=1)

predictions = np.load("y_pred_all.npy")

df['date'] = pd.to_datetime(df['date'])

df_trip_types = generate_trip_type_dataframe(df, work_cutoff)

df = df.set_index('trip_id').join(df_trip_types[['trip_id', 'trip_type']].set_index('trip_id')).reset_index()

df = standard_categories(df)

df = df.dropna(subset=['user_id', 'trip_id'])


#################### Gerando features das viagens
pool = Pool(processes=50)

mine_function = partial(retrieve_features)

trips_list = list(tqdm.tqdm(pool.imap(mine_function, df['trip_id'].unique()), total=len(df['trip_id'].unique())))

pool.close()

pool.join()

df_trips = pd.DataFrame(trips_list).sort_values(by=['user_id', 'date'])


#################### Adiciona dados das viagens anteriores feitas pelos usuários
pool = Pool(processes=50)

mine_function = partial(evaluate_user_travels)

user_trips = list(tqdm.tqdm(pool.imap(mine_function, df_trips['user_id'].unique()), total=len(df_trips['user_id'].unique())))

pool.close()

pool.join()

user_trips_df = pd.concat(user_trips).dropna()

#user_trips_df.to_csv("user_trips_table_" + str(work_cutoff) + '.csv', sep=';')

## adicionando dados da saída
if not "user_trips_table_" + str(work_cutoff) + '.csv' in os.listdir("Datasets"):

    user_trips_df.to_csv("Dataset/user_trips_table_" + str(work_cutoff) + '.csv', sep=';', index=False, header=True, mode='w')

else:

    user_trips_df.to_csv("Dataset/user_trips_table_" + str(work_cutoff) + '.csv', sep=';', index=False, header=False, mode='a')
