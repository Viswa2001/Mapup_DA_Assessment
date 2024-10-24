import pandas as pd
import numpy as np
from datetime import time

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here

    ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    n = len(ids)

    dist_matrix = pd.DataFrame(np.inf, index=ids, columns=ids)

    # Set the diagonal to 0 (distance from a node to itself)
    np.fill_diagonal(dist_matrix.values, 0)

    # Populate matrix with known distances (both directions)
    for _, row in df.iterrows():
        dist_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        dist_matrix.loc[row['id_end'], row['id_start']] = row['distance']

    for k in ids:
        for i in ids:
            for j in ids:
                dist_matrix.loc[i, j] = min(
                    dist_matrix.loc[i, j],
                    dist_matrix.loc[i, k] + dist_matrix.loc[k, j]
                )

    return dist_matrix




def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here

    unrolled_df = (
        df.stack()
        .reset_index()
        .rename(columns={'level_0': 'id_start', 'level_1': 'id_end', 0: 'distance'})
    )

    # Remove rows where id_start equals id_end (diagonal elements)
    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]

    return unrolled_df.reset_index(drop=True)

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    avg_distances = df.groupby('id_start')['distance'].mean()

    # Get the average distance for the reference ID
    ref_avg_distance = avg_distances.get(reference_id, None)
    if ref_avg_distance is None:
        return pd.DataFrame(columns=['matching_ids'])  

    # Define the threshold range (±10% of the reference ID's average distance)
    lower_bound = ref_avg_distance * 0.9
    upper_bound = ref_avg_distance * 1.1

    # Find IDs within the threshold range
    matching_ids = avg_distances[
        (avg_distances >= lower_bound) & (avg_distances <= upper_bound)
    ].index

    # Return the matching IDs as a sorted DataFrame
    return pd.DataFrame(sorted(matching_ids), columns=['matching_ids'])


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here

    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Add columns for each vehicle type by multiplying the distance with respective rates
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate

    return df

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    # Define weekday and weekend intervals with discount factors
    weekday_intervals = [
        ("Monday", "Friday", time(0, 0), time(10, 0), 0.8),
        ("Monday", "Friday", time(10, 0), time(18, 0), 1.2),
        ("Monday", "Friday", time(18, 0), time(23, 59, 59), 0.8),
    ]
    weekend_interval = ("Saturday", "Sunday", time(0, 0), time(23, 59, 59), 0.7)

    expanded_rows = []

    for _, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']

        for start_day, end_day, start_time, end_time, factor in weekday_intervals:
            row_data = {
                'id_start': id_start, 'id_end': id_end, 'distance': distance,
                'start_day': start_day, 'end_day': end_day,
                'start_time': start_time, 'end_time': end_time,
            }
            # Apply factor to each vehicle type
            for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                row_data[vehicle] = row[vehicle] * factor
            expanded_rows.append(row_data)

        start_day, end_day, start_time, end_time, factor = weekend_interval
        row_data = {
            'id_start': id_start, 'id_end': id_end, 'distance': distance,
            'start_day': start_day, 'end_day': end_day,
            'start_time': start_time, 'end_time': end_time,
        }
        # Apply factor to each vehicle type
        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            row_data[vehicle] = row[vehicle] * factor
        expanded_rows.append(row_data)

    time_based_df = pd.DataFrame(expanded_rows)
    return time_based_df
