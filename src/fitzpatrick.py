# Step 1: import pandas so we can work with CSV data
import pandas as pd
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "..", "data", "metadata.csv")
# Step 2: read the metadata.csv file from the "data" folder into a DataFrame
metadata = pd.read_csv(csv_path)

# Step 3: define a function that looks up the Fitzpatrick value
# for one or more patient IDs
def fitzpatrick(img_id):
    try:
        patient_id = img_id.split('_')[0] + '_' + img_id.split('_')[1]
    # Step 4: keep only the rows where the patient_id column
    # matches one of the IDs in the input list
        filtered = metadata[metadata["patient_id"]==(patient_id)]

    
    # Step 5: check whether any matching rows were found
        if not filtered.empty:
        
        # Step 6: take the first matching value from the
        # "fitspatrick" column and convert it to a Python integer
            return int(filtered["fitspatrick"].iloc[0])
        return np.nan
    
    except:
        # Step 7: if no match was found, return None
        return np.nan
