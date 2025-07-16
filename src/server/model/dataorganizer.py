from sklearn.model_selection import train_test_split

class DataOrganizer:

    @staticmethod
    def df_cleaner(df, cleaning_columns = None):
        columns_to_remove = DataOrganizer._find_unique_value_columns(df)
        if cleaning_columns:
            columns_to_remove += cleaning_columns
        return df[[col for col in df.columns if col not in columns_to_remove]]


    @staticmethod
    def _find_unique_value_columns(df):
        return [column for column in df.columns if df[column].nunique() == len(df)]


    @staticmethod
    def split_train_test(df):
        train_df, test_df = train_test_split(df, test_size=0.3)
        return {'train_df' : train_df, 'test_df' : test_df}


