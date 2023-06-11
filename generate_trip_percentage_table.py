import pandas as pd
import numpy as np
import tqdm


if __name__ == '__main__':

    # 1 é viagem a trabalho


    df = pd.read_csv("users_trips.csv", sep=';')

    index_predictions = pd.read_table("id_predictions.csv", sep=',').drop('Unnamed: 0', axis=1)

    predictions = np.load("y_pred_all.npy")

    df_predictions = pd.DataFrame(predictions, columns=['predictions'])

    df_predictions['review_id'] = index_predictions['review_id']

    df_trip_types = df[['trip_id', 'review_id']].set_index('review_id').join(df_predictions.set_index('review_id')).reset_index()

    qtde_places = df_trip_types['trip_id'].value_counts().reset_index().rename(columns={'index': 'trip_id', 'trip_id': 'qtde_places'})

    qtde_work = df_trip_types.groupby("trip_id").sum().reset_index().rename(columns={'predictions': 'qtde_work'})

    df_trip_types = qtde_places.set_index('trip_id').join(qtde_work.set_index('trip_id')).reset_index()

    df_trip_types['porc'] = df_trip_types['qtde_work']/df_trip_types['qtde_places']

    df_trip_types[['trip_id', 'porc', 'qtde_places', 'qtde_work']].to_csv("trip_percentage.csv", sep=';', index=False)