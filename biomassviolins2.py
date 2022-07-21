# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
# importing regex module
import re

#sns.set_style(style="whitegrid")
#sns.set_theme(style="whitegrid")
filename = 'NFHBiomassConvCompare.csv'
df =pd.read_csv(filename, dtype= int)
df.columns =df.keys()
print(df.head())
# removing null values to avoid errors 
df.dropna(inplace = True) 
#df_val= df.values*900  
# percentile list
perc =[.10, .25, .50, .75, .90]
  
# list of dtypes to include
include =['object', 'int', 'int']
  
# calling describe method
desc = df.describe(percentiles = perc, include = include)

print(desc)
#lb= df.values
#print(data_array)
#bio1 = data[1:][:1]
#LANDIS_biomass = data['LANDIS']
#Landtrendr_biomass = data['Lantrendr']
#print(bio1.head())
#file = open(filename, mode='r')
#data = file.read()
#file.close()
#dat
#print(df.data)
#totalbiomass = sns.load_dataset('totalbiomass.csv')
#ax =sns.violinplot(x=totalbiomass[data.index])

data_array=df.values[:, 1:5]

# plt.xlabel('Biomass g_C')
# plt.ylabel('iCell')
# plt.title('Above Ground Biomass Cumulative Pixel Compare')
# plt.hist(df, cumulative=True, bins=10)
# #plt.hist(df['LANDIS_B'], cumulative=True, bins=10)
# # plt.hist(df['Lantrendr01'], cumulative=True, bins=10)
# # plt.hist(df['WoodsHole '], cumulative=True, bins=10)
# #plt.legend('LANDIS_BQ', 'LANDIS_B', loc='best') #, 'LTr', 'WH', loc='best') 
# plt.show()
# plt.close()
# landis = np.array(data_array[:,0] )
# landtrendr = np.array(data_array[:,1])

# plt.xlabel('Biomass g_C')
# plt.ylabel('Count')
# plt.title('LANDIS & LANDTRENDR Above Ground Biomass Estimates')
# plt.scatter(lb.index(), lb)
# #plt.scatter(landtrendr, landtrendr)
# plt.legend(('LANDIS'), loc='best') 
# plt.show()
# # plt.close()


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Biomass Model Name')

#
# create test 
#np.random.seed(19680801)
#data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
#this is for the median and and extremes
ax1.set_title('Above Ground Biomass Model Estimates')
ax1.set_ylabel('Biomass in grams of Carbon')
ax1.violinplot(data_array)
#This is for a smoother image with those values
ax2.set_title('Above Ground Biomass Model Estimates')
parts = ax2.violinplot(
        data_array, showmeans=False, showmedians=False,
        showextrema=False)

for pc in parts['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(data_array, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data_array, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# set style for the axes
labels = ['LANDIS_B', 'LANDIS_BQ', 'LTr', 'WH']
for ax in [ax1, ax2]:
    set_axis_style(ax, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()
plt.close()