import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineering:
    def get_time_features(self, df):
        """
        Add time-based features to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with 'created_at' column.

        Returns:
            pd.DataFrame: DataFrame with added time-based features.
        """
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['month'] = df['created_at'].dt.month
        df['day'] = df['created_at'].dt.day
        df['year'] = df['created_at'].dt.year
        df['hour'] = df['created_at'].dt.hour
        df['minute'] = df['created_at'].dt.minute
        df['second'] = df['created_at'].dt.second
        df['Day_dayofweek'] = df['created_at'].dt.dayofweek
        return df

    def format_date(self, df, numeric_pattern=r'\d+'):
        """
        Format the 'created_at' column and add formatted date features.

        Args:
            df (pd.DataFrame): Input DataFrame with 'created_at' column.
            numeric_pattern (str): Regular expression pattern to extract numeric values.

        Returns:
            pd.DataFrame: DataFrame with formatted date features.
        """
        df['numeric_values'] = df['created_at'].apply(lambda x: ''.join(re.findall(numeric_pattern, x)))
        df['is_valid_date'] = pd.to_datetime(df['numeric_values'], format='%Y%m%d%H%M%S', errors='coerce').notnull()
        df = df[df['is_valid_date'] == True]
        df['datetime'] = pd.to_datetime(df['numeric_values'], format='%Y%m%d%H%M%S')
        return df

    def get_ratio_stores(self, df):
        """
        Calculate and add store-related ratio features.

        Args:
            df (pd.DataFrame): Input DataFrame with 'store_id' and 'taken' columns.

        Returns:
            pd.DataFrame: DataFrame with added ratio features.
        """
        df['orders_count'] = df.groupby('store_id')['store_id'].transform('count')
        df['stores_not_taken'] = df.groupby('store_id')['taken'].transform(lambda x: x.eq(0).sum())
        df['ratio'] = df['stores_not_taken'] / df['orders_count']
        return df

    def get_distance_ratio(self, df):
        """
        Calculate and add distance-related ratio features.

        Args:
            df (pd.DataFrame): Input DataFrame with 'to_user_distance' column.

        Returns:
            pd.DataFrame: DataFrame with added distance-related ratio features.
        """
        df['new_distancia'] = np.round(df['to_user_distance'], 1)
        df['new_distancia_count'] = df.groupby('new_distancia')['new_distancia'].transform('count')
        df['new_distancia_not_taken'] = df.groupby('new_distancia')['taken'].transform(lambda x: x.eq(0).sum())
        df['distance_cancel_ratio'] = df['new_distancia_not_taken'] / df['new_distancia_count']
        return df

    def scale_data(self, df, features=['store_id', 'to_user_distance', 'to_user_elevation',
                                       'total_earning', 'day', 'hour', 'Day_dayofweek', 'ratio']):
        """
        Scale specified features in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with features to scale.
            features (list): List of features to scale.

        Returns:
            pd.DataFrame: DataFrame with scaled features.
        """
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])
        return df


class DataProcessing:
    def select_features(self, df, features=['store_id', 'to_user_distance', 'to_user_elevation',
                                            'total_earning', 'month', 'day', 'hour', 'minute', 'Day_dayofweek',
                                            'ratio', 'distance_cancel_ratio']):
        """
        Select and return specified features from the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with features.
            features (list): List of features to select.

        Returns:
            pd.DataFrame: DataFrame with selected features.
        """
        df = df.loc[:, features]
        return df

    def process_data(self, df):
        """
        Process the input DataFrame by applying feature engineering and data processing steps.

        Args:
            df (pd.DataFrame): Input DataFrame to process.

        Returns:
            pd.DataFrame: Processed DataFrame with selected features and added ratios.
        """
        fe = FeatureEngineering()
        formatted_date = fe.format_date(df)
        data_with_time_features = fe.get_time_features(formatted_date)
        ratio = fe.get_ratio_stores(data_with_time_features)
        distance_ratio = fe.get_distance_ratio(ratio)
        scaled_data = fe.scale_data(distance_ratio)
        features = self.select_features(scaled_data)
        return features

def get_class(class_number):
    """
    Get the class label based on the class number.

    Args:
        class_number (int): Class number (1 or 0).

    Returns:
        str: Class label.
    """
    return ['Taken' if element == 1 else 'Not taken' for element in class_number]





    