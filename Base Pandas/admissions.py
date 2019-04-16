import pandas as pd 

#Reading in Data
# =============================================================================
#If I wanted to make them categories I would do this but I'm not sure if I 
#should do this since machine algorithms take numbers

# cat_vars = {'SOP': 'category', 'LOR ': 'category', 'Research':'category',
#             'University Rating': 'category'}
# df = pd.read_csv("Admission_Predict.csv", header= 0, 
#                  dtype = cat_vars)
# =============================================================================

df = pd.read_csv("Admission_Predict.csv", header= 0)
df = df.drop(columns = "Serial No.")
pd.options.display.max_columns = 20
df.info()

#Fixing Variable names 
def strip_str(x):
    x = str(x)
    x = x.replace(".", "").replace(" ", "").lower()
    return(x)
df = df.rename(strip_str, axis="columns")


#Fixing categorical variables: 
df.sop.value_counts()
df.lor.value_counts()
new_levs = {"sop": {1.5: 1, 2.5:2, 3.5:3, 4.5:4},
            "lor": {1.5: 1, 2.5:2, 3.5:3, 4.5:4}}
df = df.replace(new_levs)
df.sop.value_counts()
df.lor.value_counts()

help(pd.read_csv)



