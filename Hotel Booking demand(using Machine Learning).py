#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install plotly


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.express as px

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")

sns.set(style="whitegrid")
pd.set_option("display.max_columns", 36)


# In[3]:


hotel_booking=pd.read_csv("C:/Users/cherr/OneDrive/Documents/Desktop/hotel_bookings.csv")


# In[4]:


hotel_booking


# In[5]:


#cleaning the data
hotel_booking.isnull().sum()


# In[6]:


#lets frop the columns with high missing values
hotel_booking=hotel_booking.drop(['agent','company'],axis=1)


# In[7]:


hotel_booking


# In[8]:


#Country has 488 rows with the NaN values. 488 rows out of 119390 is negligible hence we will just remove.
hotel_booking = hotel_booking.dropna(axis = 0)


# In[9]:


hotel_booking.isnull().sum()


# In[10]:


# After cleaning, separate Resort and City hotel
# To know the acutal visitor numbers, only bookings that were not canceled are included. 
rh = hotel_booking.loc[(hotel_booking["hotel"] == "Resort Hotel") & (hotel_booking["is_canceled"] == 0)]
ch = hotel_booking.loc[(hotel_booking["hotel"] == "City Hotel") & (hotel_booking["is_canceled"] == 0)]


# In[11]:


# get number of acutal guests by country
country_data = pd.DataFrame(hotel_booking[hotel_booking["is_canceled"] == 0]["country"].value_counts())
#country_data.index.name = "country"
country_data.rename(columns={"country": "Number of Guests"}, inplace=True)
total_guests = country_data["Number of Guests"].sum()
country_data["Guests in %"] = round(country_data["Number of Guests"] / total_guests * 100, 2)
country_data["country"] = country_data.index
#country_data.loc[country_data["Guests in %"] < 2, "country"] = "Other"

# pie plot
fig = px.pie(country_data,
             values="Number of Guests",
             names="country",
             title="Home country of guests",
             template="seaborn")
fig.update_traces(textposition="inside", textinfo="value+percent+label")
fig.show()


# In[12]:


# Counting adults and children as paying guests only, not babies.
rh["adr_pp"] = rh["adr"] / (rh["adults"] + rh["children"])
ch["adr_pp"] = ch["adr"] / (ch["adults"] + ch["children"])


# In[13]:


print("""From all non-cnceled bookings, across all room types and meals, the average prices are:
Resort hotel: {:.2f} € per night and person.
City hotel: {:.2f} € per night and person."""
      .format(rh["adr_pp"].mean(), ch["adr_pp"].mean()))


# In[14]:


# normalize price per night (adr):
hotel_booking["adr_pp"] = hotel_booking["adr"] / (hotel_booking["adults"] + hotel_booking["children"])
full_data_guests = hotel_booking.loc[hotel_booking["is_canceled"] == 0] # only actual gusts
room_prices = full_data_guests[["hotel", "reserved_room_type", "adr_pp"]].sort_values("reserved_room_type")


# In[15]:


# boxplot:
plt.figure(figsize=(12, 8))
sns.boxplot(x="reserved_room_type",
            y="adr_pp",
            hue="hotel",
            data=room_prices, 
            hue_order=["City Hotel", "Resort Hotel"],
            fliersize=0)
plt.title("Price of room types per night and person", fontsize=16)
plt.xlabel("Room type", fontsize=16)
plt.ylabel("Price [EUR]", fontsize=16)
plt.legend(loc="upper right")
plt.ylim(0, 160)
plt.show()


# In[16]:


# total bookings per market segment (incl. canceled)
segments=hotel_booking["market_segment"].value_counts()

# pie plot
fig = px.pie(segments,
             values=segments.values,
             names=segments.index,
             title="Bookings per market segment",
             template="seaborn")
fig.update_traces(rotation=-90, textinfo="percent+label")
fig.show()


# In[17]:



# price per night (ADR) and person based on booking and room.
# show figure:
plt.figure(figsize=(12, 8))
sns.barplot(x="market_segment",
            y="adr_pp",
            hue="reserved_room_type",
            data=hotel_booking,
            ci="sd",
            errwidth=1,
            capsize=0.1)
plt.title("ADR by market segment and room type", fontsize=16)
plt.xlabel("Market segment", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("ADR per person [EUR]", fontsize=16)
plt.legend(loc="upper left")
plt.show()


# In[18]:


plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
sns.countplot(data = hotel_booking, x = 'stays_in_weekend_nights', hue='is_canceled', palette='rocket')
plt.title('WeekendStay vs Cancelation',fontweight="bold", size=20)

plt.subplot(1, 2, 2)
sns.countplot(data = hotel_booking, x = 'stays_in_week_nights', hue='is_canceled', palette='magma_r')
plt.title('WeekStay vs Cancelations',fontweight="bold", size=20)
plt.subplots_adjust(right=1.7)

plt.show()


# In[25]:


# Extract the data for every market segment for both city and the resort hotel

mark_seg_direct = hotel_booking[hotel_booking['market_segment'] == 'Direct']['arrival_date_year'].value_counts().resample('m').sum().to_frame()
mark_seg_online = hotel_booking[hotel_booking['market_segment'] == 'Online TA ']['arrival_date_year'].value_counts().resample('m').sum().to_frame()
mark_seg_offline = hotel_booking[hotel_booking['market_segment'] == 'Offline TA/TO']['arrival_date_year'].value_counts().resample('m').sum().to_frame()
mark_seg_groups = hotel_booking[hotel_booking['market_segment'] == 'Groups']['arrival_date_year'].value_counts().resample('m').sum().to_frame()
mark_seg_corporate = hotel_booking[hotel_booking['market_segment'] == 'Corporate']['arrival_date_year'].value_counts().resample('m').sum().to_frame()
mark_seg_complementary = hotel_booking[hotel_booking['market_segment'] == 'Complementary']['arrival_date_year'].value_counts().resample('m').sum().to_frame()
mark_seg_aviation = hotel_booking[hotel_booking['market_segment'] == 'Aviation']['arrival_date_year'].value_counts().resample('m').sum().to_frame()


################################################################### SUBPLOT ACROSS MARKET SEGMENT ##################################################################################

fig = go.Figure()
fig.add_trace(go.Scatter(x=mark_seg_direct.index, y=mark_seg_direct['arrival_date_year'], name="Direct"))
fig.add_trace(go.Scatter(x=mark_seg_online.index, y=mark_seg_online['arrival_date_year'], name="Online TA"))
fig.add_trace(go.Scatter(x=mark_seg_offline.index, y=mark_seg_offline['arrival_date_year'], name="Offline TA/TO"))
fig.add_trace(go.Scatter(x=mark_seg_groups.index, y=mark_seg_groups['arrival_date_year'], name="Groups"))
fig.add_trace(go.Scatter(x=mark_seg_corporate.index, y=mark_seg_corporate['arrival_date_year'], name="Corporate"))
fig.add_trace(go.Scatter(x=mark_seg_complementary.index, y=mark_seg_complementary['arrival_date_year'], name="Complementary"))
fig.add_trace(go.Scatter(x=mark_seg_aviation.index, y=mark_seg_aviation['arrival_date_year'], name="Aviation"))
fig.update_layout(title_text='Total Monthly Bookings Across Market Segments', title_x=0.5, title_font=dict(size=20))  
fig.update_layout(xaxis_title="arrival_date_year",yaxis_title="Total Bookings")
fig.show()


# In[26]:


# `Meal` feature donut chart

meal_labels= ['BB','HB', 'SC', 'Undefined', 'FB']
size = hotel_booking['meal'].value_counts()
plt.figure(figsize=(10,10))
cmap =plt.get_cmap("Pastel2")
colors = cmap(np.arange(3)*4)
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(size, labels=meal_labels, colors=colors, wedgeprops = { 'linewidth' : 5, 'edgecolor' : 'white' })
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Meal Types', weight='bold')
plt.show()


# In[31]:


a = hotel_booking.corr()
plt.figure(figsize=(12, 10))
k = 15
cols = a.nlargest(k,'is_canceled')['is_canceled'].index
cm = np.corrcoef(hotel_booking[cols].values.T)
sns.set(font_scale=1.0)
hm = sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels=cols.values,xticklabels=cols.values,cmap="Blues")
plt.show()


# In[32]:


plt.figure(figsize = (24, 12))

corr = hotel_booking.corr()
sns.heatmap(corr, annot = True, linewidths = 1)
plt.show()


# In[37]:


keep_list=['hotel is_canceled', 'lead_time', 
           'stays_in_weekend_nights', 
           'stays_in_week_nights', 'adults', 'children', 'adr', 'arrival_date_day_of_month']
for col in hotel_booking.columns:
     if col not in keep_list:
            del hotel_booking[col]


# In[38]:


hotel_booking.head()


# In[39]:


f,ax = plt.subplots(figsize=(16, 9))
plt.xticks(rotation=45,fontsize=15 );
plt.yticks(fontsize=15);
sns.heatmap(hotel_booking.corr(), annot=True, linewidths=.5, ax=ax)
sns. set(font_scale=2)


# In[41]:


# Correlation Matrix with Spearman method

plt.figure(figsize=(15,15))
corr_categorical=hotel_booking.corr(method='spearman')
mask_categorical = np.triu(np.ones_like(corr_categorical, dtype=np.bool))
sns.heatmap(corr_categorical, annot=True, fmt=".2f", cmap='BrBG', vmin=-1, vmax=1, center= 0,
            square=True, linewidths=2, cbar_kws={"shrink": .5}).set(ylim=(15, 0))
plt.title("Correlation Matrix Spearman Method- Categorical Data ",size=15, weight='bold')


# In[ ]:




