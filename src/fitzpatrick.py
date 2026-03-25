# Step 1: import pandas so we can work with CSV data
import pandas as pd

# Step 2: read the metadata.csv file from the "data" folder into a DataFrame
metadata = pd.read_csv("data/metadata.csv")

# Step 3: define a function that looks up the Fitzpatrick value
# for one or more patient IDs
def fitzpatrick(patient_ids):
    
    # Step 4: keep only the rows where the patient_id column
    # matches one of the IDs in the input list
    filtered = metadata[metadata["patient_id"].isin(patient_ids)]
    
    # Step 5: check whether any matching rows were found
    if not filtered.empty:
        
        # Step 6: take the first matching value from the
        # "fitspatrick" column and convert it to a Python integer
        return int(filtered["fitspatrick"].iloc[0])
    
    else:
        # Step 7: if no match was found, return None
        return None

# Step 8: call the function with a patient ID
result = fitzpatrick(["PAT_756"])

# Step 9: print the result
print(result)