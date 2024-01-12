import os
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def get_region(state):
    northeast_states = ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont", "New Jersey", "New York", "Pennsylvania"]
    midwest_states = ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska", "North Dakota", "South Dakota"]
    south_states = ["Delaware", "Florida", "Georgia", "Maryland", "North Carolina", "South Carolina", "Virginia", "West Virginia", "Alabama", "Kentucky", "Mississippi", "Tennessee", "Arkansas", "Louisiana", "Oklahoma", "Texas"]
    west_states = ["Arizona", "Colorado", "Nevada", "New Mexico", "Utah", "Wyoming", "Alaska", "California", "Hawaii", "Oregon", "Washington"]

    if state in northeast_states:
        return "North East"
    elif state in midwest_states:
        return "Midwest"
    elif state in south_states:
        return "South"
    elif state in west_states:
        return "West"
    else:
        return "Unknown"

def process_dataset(file_path):
    # Load the dataset into a pandas DataFrame
    df = load_data(file_path)

    # Add the "Region" field based on the "Location" field
    df['Region'] = df['Location'].apply(get_region)

    # Define age group bins
    age_bins = [0, 18, 27, 36, 45, 55, float('inf')]
    age_labels = ['0-18', '19-27', '28-36', '37-45', '46-55', '55+']

    # Add the "Age Group" field based on the "Age" field
    df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

    # Remove specified fields
    fields_to_remove = ['Customer ID', 'Age', 'Purchase Amount (USD)', 'Location', 'Size', 'Review Rating', 'Subscription Status', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Previous Purchases', 'Payment Method', 'Frequency of Purchases']
    df = df.drop(fields_to_remove, axis=1)

    return df