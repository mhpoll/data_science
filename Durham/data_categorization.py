# %%
import pandas as pd
import numpy as np
from openpyxl import load_workbook

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# %%
df=pd.read_excel('./Dummy Data for Python.xlsx')

# %%
df.head()

# %%
print(df['File Type'].unique())
print(df['last_modified'].unique())

# %%
df_2021_2024=df.loc[(df['last_modified']>=2021) & (df['last_modified']<=2024)]
print(df_2021_2024['last_modified'].shape)
print(df_2021_2024['last_modified'].unique())

# %%
df_2015_2020=df.loc[(df['last_modified']>=2015) & (df['last_modified']<=2020)]
print(df_2015_2020['last_modified'].shape)
print(df_2015_2020['last_modified'].unique())

# %%
df_pre_2015=df.loc[(df['last_modified']<2015)]
print(df_pre_2015['last_modified'].shape)
print(df_pre_2015['last_modified'].unique())

# %%
def categorization(df, year_column, sum_column, file_type_column):
    conditions = [
        (df[year_column] >= 2021) & (df[year_column] <= 2024),
        (df[year_column] >= 2015) & (df[year_column] <= 2020),
        (df[year_column] < 2015)
    ]
    ranges = ["2021-2024", "2015-2020", "pre-2015"]

    data = {
        "Year_Range": [],
        "File_Count": [],
        "Sum_File_Size(mb)": [],
        "File_Type_Counts": []
    }

    for i, condition in enumerate(conditions):
        df_filtered = df[condition]

        file_count = df_filtered.shape[0]
        sum_column_value = df_filtered[sum_column].sum()

        file_type_count = df_filtered[file_type_column].value_counts().to_dict()

        data["Year_Range"].append(ranges[i])
        data["File_Count"].append(file_count)
        data["Sum_File_Size(mb)"].append(sum_column_value)
        data["File_Type_Counts"].append(file_type_count)

    range_stats_df = pd.DataFrame(data)

    return range_stats_df

# %%
new_df=categorization(df,'last_modified','size_mb','File Type')

# %%
new_df

# %%
path='./new_excel_file.xlsx'


writer = pd.ExcelWriter(path, engine='openpyxl')

new_df.to_excel(writer,sheet_name='data_categorization')
df_pre_2015.to_excel(writer, sheet_name='pre_2015')
df_2015_2020.to_excel(writer, sheet_name='2015-2020')
df_2021_2024.to_excel(writer,sheet_name='2021-2024')

writer.close()



# %%
