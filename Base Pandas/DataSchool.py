import pandas as pd 

df = pd.read_csv("Admission_Predict.csv", header= 0)
df.head()
pd.options.display.max_columns = 20

#Fixing Variable names 
def strip_str(x):
    x = str(x)
    x = x.replace(".", "").replace(" ", "").lower()
    return(x)
df = df.rename(strip_str, axis="columns")

#Fixing categorical variables: 
new_levs = {"sop": {1.5: 1, 2.5:2, 3.5:3, 4.5:4},
            "lor": {1.5: 1, 2.5:2, 3.5:3, 4.5:4}}
df = df.replace(new_levs)





