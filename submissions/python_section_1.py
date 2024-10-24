from typing import Dict, List

import pandas as pd
import re
import math
import polyline

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    def reverse_sublist(lst: List[int]) -> List[int]:
        n = len(lst)
        p1 = 0
        p2 = n-1
        while p1<p2:
            lst[p1],lst[p2] = lst[p2],lst[p1]
            p1+=1
            p2-=1
        return lst
        
    resultant_list = []
    for i in range(0,len(lst),n):
        group = lst[i:i+n]
        resultant_list.extend(reverse_sublist(group))
    
    return resultant_list


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    my_dict = {}
    for string in lst:
        if len(string) in my_dict:
            my_dict[len(string)].append(string)
        else:
            my_dict[len(string)]=[string]
            
    return dict(sorted(my_dict.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    def _flatten(current, parent_key=""):
        flattened = {}
        for key, value in current.items():
            
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                # Recursively flatten the nested dictionary
                flattened.update(_flatten(value, new_key))
            elif isinstance(value, list):
                # Flatten list elements with their indices
                for idx, item in enumerate(value):
                    indexed_key = f"{new_key}[{idx}]"
                    if isinstance(item, dict):
                        flattened.update(_flatten(item, indexed_key))
                    else:
                        flattened[indexed_key] = item
            else:
                flattened[new_key] = value

        return flattened

    # Start the flattening process
    return _flatten(nested_dict)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        
        used = set()
        for i in range(start, len(nums)):
            if nums[i] in used:
                continue
            used.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]
    
    result = []
    nums.sort()  
    backtrack(0)
    return result


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    pattern = re.compile(r'''
        \b                  
        (
            (?:0[1-9]|[12][0-9]|3[01])  # Day: 01-31
            -
            (?:0[1-9]|1[0-2])           # Month: 01-12
            -
            \d{4}                       # Year: Any four digits
        |
            (?:0[1-9]|1[0-2])           # Month: 01-12
            /
            (?:0[1-9]|[12][0-9]|3[01])  # Day: 01-31
            /
            \d{4}                       # Year: Any four digits
        |
            \d{4}                       # Year: Any four digits
            \.
            (?:0[1-9]|1[0-2])           # Month: 01-12
            \.
            (?:0[1-9]|[12][0-9]|3[01])  # Day: 01-31
        )
        \b                  
    ''', re.VERBOSE)

    dates = re.findall(pattern,text)

    return dates



def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.

    """
    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the Haversine distance between two points on Earth.
        Parameters are given in decimal degrees.
        Returns the distance in meters.
        """
        R = 6371000  # Radius of the Earth in meters

        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Compute differences
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c
    
    coordinates = polyline.decode(polyline_str)

    # Create DataFrame with latitude and longitude columns
    df = pd.DataFrame(coordinates, columns=["latitude", "longitude"])

    # Initialize the distance column with 0 for the first row
    df["distance"] = 0.0

    # Calculate distances using Haversine formula
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, ["latitude", "longitude"]]
        lat2, lon2 = df.loc[i, ["latitude", "longitude"]]
        df.loc[i, "distance"] = haversine(lat1, lon1, lat2, lon2)

    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)
    
    # Step 1: Rotate the matrix 90 degrees clockwise
    rotated_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n-1-i] = matrix[i][j]
    
    # Step 2: Calculate sums and replace elements
    final_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    # Drop rows with missing values in critical columns
    df = df.dropna(subset=['startDay', 'startTime', 'endDay', 'endTime'])
    
    # Convert start and end times to datetime
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')
    
    # Drop rows where conversion failed
    df = df.dropna(subset=['start', 'end'])
    
    # cases where end time is earlier than start time
    df.loc[df['end'] < df['start'], 'end'] += pd.Timedelta(days=1)
    
    # Sort the dataframe by id, id_2, and start time
    df = df.sort_values(['id', 'id_2', 'start'])
    
    # Group by id and id_2
    grouped = df.groupby(['id', 'id_2'])
    
    def check_group(group):
        # Check if all days are covered
        days_covered = set(group['startDay'].unique()) | set(group['endDay'].unique())
        all_days_covered = len(days_covered) == 7
        
        # Check if full 24 hours are covered
        intervals = list(zip(group['start'], group['end']))
        
        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
        
        # Check if merged intervals cover 24 hours
        total_coverage = sum((interval[1] - interval[0]).total_seconds() for interval in merged)
        full_coverage = total_coverage >= 24 * 60 * 60  
        
        # Return True if either condition is not met (incorrect timestamps)
        return not (all_days_covered and full_coverage)
    
    result = grouped.apply(check_group)
    
    return result
