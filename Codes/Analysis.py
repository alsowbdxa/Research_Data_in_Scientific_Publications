import gc
import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import random
from pandas.core.frame import DataFrame
import pickle
import sqlite3
tqdm.pandas()

d = pd.read_csv('sample_data.csv') # this is a sample of our dataset, the whole data can be download from PMC (https://pmc.ncbi.nlm.nih.gov/tools/openftlist/) and follow the methods we used in the paper.

# Plot 1
keyword_counts = d['keyword'].value_counts()
top_keywords = keyword_counts.head(10)
# 
plt.figure(figsize=(10, 6),dpi=600)
top_keywords[::-1].plot(kind='barh')
plt.xlabel('Count')
plt.ylabel('Keyword')
plt.show()

# Plot 2
value_counts = d['predicted_label_str'].value_counts()
# 
plt.figure(figsize=(10, 6),dpi=600)
sns.barplot(x=value_counts.values, y=value_counts.index)
plt.xlabel('Count')
plt.ylabel('')
plt.show()

# Plot 3
top_keywords = d['keyword'].value_counts().head(20).index
filtered_df = d[d['keyword'].isin(top_keywords)]
percentage_df = (filtered_df.groupby(['keyword', 'predicted_label_str']).size() /
                 filtered_df.groupby('keyword').size())
percentage_df = percentage_df.reset_index(name='percentage')
merged_df = percentage_df.pivot(index='keyword', columns='predicted_label_str', values='percentage').reset_index()
merged_df.sort_values('keyword')
merged_df['keyword'] = pd.Categorical(merged_df['keyword'], categories=top_keywords, ordered=True)
merged_df.sort_values('keyword',ascending=False,inplace=True)
merged_df = merged_df[['keyword', 'release','reuse','reference','nothing']]

plt.rcParams["figure.dpi"] = 600
merged_df.plot(x='keyword', kind='barh', stacked=True)
plt.ylabel('Repository link')
plt.xlabel('Percentage')
plt.legend(title='Predicted Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Plot 4
intention_colors = {
    'release': '#1f77b4',     
    'reuse': '#ff7f0e',       
    'reference': '#2ca02c' 
}

def explode_with_weights(row):
    categories = row['category']
    weight = 1 / len(categories) 
    return pd.DataFrame({
        'category': categories,
        'predicted_label_str': row['predicted_label_str'],
        'weight': weight
    })

df_expanded = pd.concat([explode_with_weights(row) for _, row in d.iterrows()], ignore_index=True)

grouped = df_expanded.groupby(['category', 'predicted_label_str']).agg({'weight': 'sum'}).reset_index()
normalized = grouped.groupby('category').apply(lambda x: x.assign(weight=x['weight'] / x['weight'].sum())).reset_index(drop=True)
pivot_table = normalized.pivot(index='category', columns='predicted_label_str', values='weight').fillna(0)
pivot_table = pivot_table[['release', 'reuse', 'reference']]
pivot_long = pivot_table.reset_index().melt(id_vars='category', var_name='Intention', value_name='Proportion')

plt.figure(figsize=(12, 6))
sns.set(style='whitegrid')
bottoms = pd.Series(0, index=pivot_table.index) 
for intention in ['release', 'reuse', 'reference']:
    sns.barplot(
        x='Proportion', y='category', data=pivot_long[pivot_long['Intention'] == intention],
        label=intention, color=intention_colors[intention], left=bottoms
    )
    bottoms += pivot_table[intention]
plt.xlabel('Proportion')
plt.ylabel('Discipline')
plt.legend(title='Intention', loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.show()

# Plot 5
total_per_year = d.groupby('pub_year').size().rename('total')  
df_prop = d.groupby(['pub_year', 'predicted_label_str']).size().reset_index(name='count')
df_prop = df_prop.merge(total_per_year, on='pub_year')
df_prop['percentage'] = df_prop['count'] / df_prop['total'] * 100

custom_palette = {
    'release': '#1f77b4',  
    'reuse': '#ff7f0e',    
    'reference': '#2ca02c',
    'nothing': '#d62728'  
}

fig, ax1 = plt.subplots(figsize=(14, 7), dpi=600) 
sns.barplot(data=df_prop, x='pub_year', y='percentage', hue='predicted_label_str', hue_order=[ 'release','reuse','reference'],ax=ax1, palette=custom_palette)
# ax1.set_title('Distribution of Intention Over Time')
ax1.set_ylabel('Percentage')
ax1.set_xlabel('Publication year')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
plt.legend(title="Intention", loc='upper left',bbox_to_anchor=(0.06, 1))
plt.tight_layout()
plt.show()















