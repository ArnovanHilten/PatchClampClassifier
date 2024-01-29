def process_metadata(metadata, savepath = 'drive/MyDrive/Postdoc/data_juliet/'):

    metadata_df = metadata.ffill()
    metadata_df

    # Forward filling the 'Date/filename' column and replacing "/" with "_"
    metadata_df['Date/filename'] = metadata_df['Date/filename'].fillna(method='ffill')
    metadata_df['Date_filename_modified'] = metadata_df['Date/filename'].str.replace('/', '_')

    # Changing the type of 'Date/filename' column to date
    metadata_df['Date/filename'] = pd.to_datetime(metadata_df['Date/filename'], format='%d/%m/%Y')
    metadata_df["Class"] = (metadata_df["Input"] == "Poly")*1
    # Attempting again to reformat 'Date/filename' to 'year_month_day' format for 'Date_filename_modified'
    metadata_df['Date_filename_modified'] = metadata_df['Date/filename'].dt.strftime('%Y_%m_%d')
    metadata_df.head()

    metadata_df["filename"] = metadata_df["Date_filename_modified"].values + "_" + metadata_df["recordings"].str.replace("rec","0").values
    metadata_df
    if savepath is not(None):
        metadata_df.to_csv("metadata.csv", index=False)
    return metadata_df