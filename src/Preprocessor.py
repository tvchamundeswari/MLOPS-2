import pandas as pd


def load_data(input_file="src/loan_data.csv", output_file="src/loan_data_cleaned.csv", columns_to_drop=None):
    """
    Load the CSV file, drop specified columns, and save the cleaned data.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the cleaned CSV file.
        columns_to_drop (list): List of column names to be dropped.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """

    try:
        # Load the CSV file into a Pandas DataFrame
        df = pd.read_csv(input_file)

        # Columns to drop (if provided)
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors="ignore")

        # Save the modified DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")

        return df

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return None


if __name__ == "__main__":
    # Define columns to drop
    columns_to_remove = [
        'Dependents', 'Education', 'Self_Employed', 'CoapplicantIncome',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]

    # Load and preprocess data
    processed_df = load_data(columns_to_drop=columns_to_remove)
