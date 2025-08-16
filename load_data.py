import pandas as pd # type: ignore

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    :param file_path: Path to the CSV file
    :return: DataFrame containing the data
    """
    try:
        df = pd.read_csv(file_path)
        #print("Columns : ", df.columns.tolist())
        # print("Data Types : ", df.dtypes)
        # print("First 5 rows of the DataFrame:")
        # print(df.head())
        #print(df["decision_type"].unique())  # Print unique values in 'decision_type' column


        return df

    except Exception as e :
        print(f"Error loading data from {file_path}: {e}")
        return None

if __name__ == "__main__":
    file_path = "C:\\Users\\varsh\\Downloads\\justice_sample.csv"
    load_data(file_path)