#!/usr/bin/env python
# coding: utf-8

# # Hello! Welcome to my travel events timeline; I wanted to plot some travels that are memorable to me, like moving and personal interests!
#     :) 

# In[29]:


# imports plotyly
import plotly.express as px
import pandas as pd


# In[76]:


# my events 
events = {
    'Event': [
        'Birthplace: Cebu, PH',
        'Moved to Maryland, USA', 
        'Moved to Zuni Native American Reservation, USA',
        'Moved back to Cebu, PH', 
        'Moved to New Mexico', 
        'Trip to Germany',
        'Trip to Czech Republic',
        'Trip to Slovakia Republic',
        'Trip to Hungary',
        'Trip to Poland',
        'Trip to Austria',
        'Return to USA from Netherlands',
        'First domestic flight in the U.S. to West coast',
        '30hr Road trip across major cities in Canada'
    ],
    'Date': [
        '2000-08-19', 
        '2007-09-15', 
        '2011-05-01', 
        '2014-10-15', 
        '2015-05-15', 
        '2018-06-09',
        '2018-06-11',
        '2018-06-12',
        '2018-06-15',
        '2018-06-17',
        '2018-06-17',
        '2018-06-20',
        '2021-07-12',
        '2024-07-08'
    ]
}


# In[77]:


# Creates a data frame
df = pd.DataFrame(events)

# Converts date to datetime
df['Date'] = pd.to_datetime(df['Date'])


# In[78]:


# Creates a timeline plot
fig = px.scatter(df, x='Date', y='Event', color='Event', title='Life Events Timeline', size=[15]*len(df))

# Customizes the layout
fig.update_traces(marker=dict(line=dict(width=2, color='pink')))
fig.update_layout(
    yaxis_title='Event', 
    xaxis_title='Date',
    showlegend=False, 
    title_font=dict(size=24, color='pink'),  # Title font
    font=dict(family='Baskerville', size=12, color='purple'), # General font
    plot_bgcolor='pink',
    xaxis=dict(showgrid=False),  # Grid customization
    yaxis=dict(showgrid=False)
)


# Show the plot
fig.show()

