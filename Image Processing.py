#%%Importing Libraries
import os
import pandas as pd
import re
import numpy as np
#%%Renaming Images to indentify tumor or no tumor
data_csv=pd.read_csv("Brain_Tumor.csv")
folder = r"C:\Users\pavan\Desktop\Git\MyWork\Brain Tumor Images\\"
folder1 = r"C:\Users\pavan\Desktop\Git\MyWork\Brain Tumor Images Renamed\Tumor\\"
folder0 = r"C:\Users\pavan\Desktop\Git\MyWork\Brain Tumor Images Renamed\No Tumor\\"
count = 1
# count increase by 1 in each iteration
# iterate all files from a directory
for file_name in os.listdir(folder):
    
    source = folder + file_name
    #Target value Tumor = 1 Non tumor =0
    v=data_csv.iloc[count-1,1]
    if v == 1:
        v = str("Tumor")
        destination = folder1 + "image"+ str(count) + "_" + str(v)+ ".jpg"

    else:
        v = str("No Tumor")
        destination = folder0 + "image"+ str(count) + "_" + str(v)+ ".jpg"
    
    # Renaming the file
    os.rename(source, destination)
    count += 1
print('All Files Renamed')

print('New Names are')
# verify the result
res = os.listdir(folder)
print(res)   


 
#%%   

