import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r'C:\Users\reece\Desktop\Python_Scripts\Base_Matplotlib')

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


plt.scatter(df.grescore, df.cgpa)
plt.show()
