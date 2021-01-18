#!/usr/bin/env python
# coding: utf-8

# Consumer API keys
# JICrJp34UA3V9qb0RliDWljTC (API key)
# 
# XVcmq6tKOFEaYrvm2bWPL3SynVaOnDAzHBtNRopDjyoCmAYRPa (API secret key)
# 
# Regenerate
# Access token & access token secret
# 1080314182577225731-7XBMA10n8DfxCk2dkDUfsAkxMdnrsR (Access token)
# 
# TgpOLhoDyj2dQQeUyjkchxMJ5FQeOeCaJvOukst90KBdP (Access token secret)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import tweepy
import json
import re
import datetime as dt
from timeit import default_timer as timer


# In[2]:


#Read twitter-archive csv file as a pandas dataframe and check its structure

twitter_archive = pd.read_csv('twitter-archive-enhanced.csv')
twitter_archive.head()


# In[3]:


#Tsv file will be downloaded programatically using Reeuests library

url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'

response = requests.get(url)

#Save tsv to file

with open('image_predictions.tsv',mode='wb') as file:
    file.write(response.content)
    
#Read in tsv file in pandas dataframe
image_predictions = pd.read_csv('image_predictions.tsv', sep='\t')
image_predictions.head()


# #Twitter API Authentical Details
# 
# consumer_key = 'YOUR CONSUMER KEY'
# consumer_secret = 'YOUR CONSUMER SECRET'
# access_token = 'YOUR ACCESS TOKEN'
# access_secret = 'YOUR ACCESS SECRET'
# 
# #Twitter Variables
# auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
# auth.set_access_token(access_token,access_secret)
# 
# api = tweepy.API(auth,wait_on_rate_limit=True)

# #Twitter API Authentical Details
# 
# consumer_key = 'JICrJp34UA3V9qb0RliDWljTC' 
# 
# consumer_secret = 'XVcmq6tKOFEaYrvm2bWPL3SynVaOnDAzHBtNRopDjyoCmAYRPa'
# 
# access_token = '1080314182577225731-7XBMA10n8DfxCk2dkDUfsAkxMdnrsR'
# 
# access_secret = 'TgpOLhoDyj2dQQeUyjkchxMJ5FQeOeCaJvOukst90KBdP'
# 
# #Twitter Variables
# auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
# auth.set_access_token(access_token,access_secret)
# 
# api = tweepy.API(auth,wait_on_rate_limit=True)

# In[5]:


#Add each tweet to a new line of tweet_json.txt
start = timer()
with open('tweet_json.txt','w', encoding='utf8') as tweet_data:
    #This loop will likely take 20-30 minutes to run because of Twitter's rate limit for tweet_id in tweet_ids:
    for tweet_id in twitter_archive['tweet_id']:
        
        try:
            tweet = api.get_status(tweet_id, tweet_mode = 'extended')
            json.dump(tweet._json,tweet_data)
            tweet_data.write('\n')
            
        except:
            continue
            
end = timer()
print(end-start)


# In[4]:


#Append each tweet into a list

tweets_data = []

tweet_file  = open('tweet_json.txt','r')

for line in tweet_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        print('error')
        continue
        
tweet_file.close()


# In[5]:


#Create dataframe for tweet information
tweet_info = pd.DataFrame()

#Add variables to df: tweet IF, retweet count, favorite count

tweet_info['tweet_id'] = list(map(lambda x: x['id'],tweets_data))

tweet_info['retweet_count'] = list(map(lambda x: x['retweet_count'],tweets_data))

tweet_info['favorite_count'] = list(map(lambda x: x['favorite_count'],tweets_data))

tweet_info.head()


# In[6]:


tweet_info.head()


# ## Assess
# 
# Three DataFrames that will be assessed in the next section.
# 
# 1) twitter_archive - This dataframe has retweet and favorite counts. 
# 
# 2) image_predictions - This dataframe has the results of neural network trying to identify dog breed in a tweet's picture.
# 
# 3) tweet_info - This dataframe has tweet's text, rating and dog category.
# 

# In[7]:


twitter_archive.info()


# There are 2356 datapoints in twitter_archive file. There are missing values for following columns: in_reply_to_status_id and in_reply_to_user_id, retweet_status_id, retweeted_status_user_id, retweeted_status_timestamp and expanded urls. The data type for timestamp is not in datetime format. The numerator and denominator rating is in integer, instead of float.

# In[8]:


twitter_archive.rating_denominator.value_counts()


# In[9]:


twitter_archive.rating_numerator.value_counts()


# In[10]:


#Make sure all id's are unique, no duplicates
twitter_archive[twitter_archive.tweet_id.duplicated()]


# In[11]:


twitter_archive.describe()


# The maximum value of 1776 seems to be a outlier for the numerator rating. Also, for the rating_denominator, the maximum value of 170 seems to be a outlier. 

# In[12]:


np.count_nonzero(twitter_archive['rating_numerator'] > 10) , np.count_nonzero(twitter_archive['rating_numerator'] > 20)


# There are about 1455 dogs which are rated greater than 10, while only 24 dogs with rating above 20.

# In[13]:


twitter_archive['name'].value_counts()


# ### Assess: image_predictions 
# 
# 

# In[14]:


image_predictions.info()


# There are about 2075 entries with no missing values.

# In[15]:


image_predictions.head(3)


# Here, each column represents as following:
#     
#     1) tweet_id: unique tweet identifier
#     2) jpg_url: image of the dog
#     3) img_num: possible image number of 4 possible images
#     4) p1: the algorithm's first prediction for the image in the tweet
#     5) p1_conf: how much likely the first prediction is true
#     6) p1_dog: true or false if the first prediction is a breed of dog
#     7) p2: the algorithm's second predcition for the image in the tweet
#     8) p2_conf: how much likely the second prediction is true
#     9) p2_dog: true or false if the second prediction is a breed of dog
#     10) p3: the algorithm's third predcition for the image in the tweet
#     11) p3_conf: how much likely the third prediction is true
#     12) p3_dog: true or false if the second prediction is a breed of dog

# In[16]:


image_predictions['img_num'].value_counts()


# From the above value counts of 'img_num', one can see that the first prediction of the algorithm seems to be the most possible image.

# In[17]:


#How many first predictions are actually dog
image_predictions['p1_dog'].value_counts()


# In[18]:


#How many second predictions are actually dog
image_predictions['p2_dog'].value_counts()


# In[19]:


#How many third predictions are actually dog
image_predictions['p3_dog'].value_counts()


# In[20]:


#Find rows where p1, p2, p3 are all false

(image_predictions[(image_predictions['p1_dog']==False) & 
(image_predictions['p2_dog']==False) & (image_predictions['p3_dog']==False)]).count()


# There are 324 entries, which do not represent tweet about a dog.

# In[21]:


image_predictions['tweet_id'].duplicated().sum()


# There are no dulplicate tweet_id.

# ### Assess: tweet_info

# In[22]:


tweet_info.info()


# There are 2342 entries with no missing values. The tweet_info dataframe has the retweet count and favorite count assosciated with each tweet_id.

# In[23]:


tweet_info.head()


# In[24]:


#Check if no tweet id's are duplicates

tweet_info['tweet_id'].duplicated().sum()


# ### Assess Summary
# 
# #### Quality:
# 
# 1)twitter_archive, tweet_id datatype is an integer. It should be converted to string.
# 
# 2)twitter_archive has 181 rows in the retweet column. We only want the original ratings, no retweets.
# 
# 3)twitter_archive contains some columns, that are not needed for our analysis. Therefore, they can be removed. These columns are as followings: "in_reply_to_status_id", "in_reply_to_user_id", "retweeted_status_id", "retweeted_status_user_id", "retweeted_status_timestamp".
# 
# 4)twitter_archive has timestamp column not in datetime datatype. 
# 
# 5)twitter_archive, "source" column consist of url. Instead, it is better to convert the url to text. So it becomes easier to read the source.
# 
# 6)twitter_archive, "rating_numerator" and "rating_denominator"  columns consist of outliers. Also, their data types should be converted from integer to float.
# 
# 7)twitter_archive, "name", all names should have consistent formating. Therefore, the first letter of the name should be capitalized. Also, some names are not dog names such as "None" and "a". These names should be removed.
# 
# 8)image_predictions, the rows where p1,p2 and p3 are all false, these rows can be removed from the dataframe.
# 
# 9)image_predictions, the columns names, p1, p2, p3 are not informative. There should be more description column names.
# 
# 
# #### Tidiness: 
# 
# 1)twitter_archive: There are various dog stage columns such as doggo, floofer, etc. Instead, it is better to create a column, "dog stage" by extracting values from these four columns (doggo, floofer, pupper, puppo). If none valules are present in all three columns, then it should be represented as "Null".
# 
# 2)Three dataframes: twitter_archive, image_predictions, and tweet_info should be joined into one master dataset.

# ## Clean

# Before preceding with the cleaning, it is a good idea to make copies of the original files.

# In[25]:


#Copy all three datasets

twitter_archive_clean = twitter_archive.copy()

image_predictions_clean = image_predictions.copy()

tweet_info_clean = tweet_info.copy()


# #### Clean: Quality 1
# 
# __Define:__ Remove rows with "retweeted_status_x' since we are only interested in original tweets only. 
# 
# __Code:__

# In[26]:


twitter_archive_clean = twitter_archive_clean[twitter_archive_clean['retweeted_status_id'].isnull()]


# __Test__: The number of entries should decrease by 181. So from the original length of 2356, the total number of entries for tweet_id variable should be 2175.

# In[27]:


twitter_archive_clean.info()


# #### Clean: Quality 2

# __Define__: Drop the following columns from twitter_archive: "in_reply_to_status_id", "in_reply_to_user_id", "retweeted_status_id", "retweeted_status_user_id", "retweeted_status_timestamp".
# 
# __Code__:

# In[28]:


twitter_archive_clean.drop(["in_reply_to_status_id", "in_reply_to_user_id", 
            "retweeted_status_id", "retweeted_status_user_id", "retweeted_status_timestamp"], axis=1, inplace=True)


# __Test__: Check if the columns mentioned in the code are dropped or not.

# In[29]:


twitter_archive_clean.info()


# #### Clean: Quality 3

# __Define__: Convert the twitter_archive timestamp to datetime. Then parse the column into date and time.
# 
# __Code__:

# In[30]:


twitter_archive_clean ['timestamp'] = pd.to_datetime(twitter_archive_clean['timestamp'])


# In[31]:


# Convert timestamp to datetime
from datetime import date as dt
twitter_archive_clean['Day'] = twitter_archive_clean ['timestamp'].dt.day
twitter_archive_clean['Month'] = twitter_archive_clean ['timestamp'].dt.month
twitter_archive_clean['Year'] = twitter_archive_clean ['timestamp'].dt.year
twitter_archive_clean['Time'] = twitter_archive_clean ['timestamp'].dt.time

# Create day of week column
twitter_archive_clean['weekday'] = twitter_archive_clean['timestamp'].dt.dayofweek
days = {0:'Mon',1:'Tues',2:'Weds',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'}
twitter_archive_clean['weekday'] = twitter_archive_clean['weekday'].apply(lambda x: days[x])


# __Test__: 

# In[32]:


#Check if the Day, Month, Year, Time and weekday
twitter_archive_clean.info()


# In[33]:


twitter_archive_clean.head(3)


# #### Clean: Quality 4

# __Define:__ Convert url to text from Source column of the twitter_archive dataframe.
# 
# __Code:__

# In[34]:


# Replacing the text with urls
source_text = {'<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>': 'Twitter for iPhone',
 '<a href="http://vine.co" rel="nofollow">Vine - Make a Scene</a>': 'Vine - Make a Scene',
 '<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>': 'Twitter Web Client',
 '<a href="https://about.twitter.com/products/tweetdeck" rel="nofollow">TweetDeck</a>': 'TweetDeck'}


# In[35]:


# Function to apply
def abbreviate_source(twitter_archive_clean):
    if twitter_archive_clean['source'] in source_text.keys():
        abbrev = source_text[twitter_archive_clean['source']]
        return abbrev
    else:
        return twitter_archive_clean['source']
    
twitter_archive_clean['source'] = twitter_archive_clean.apply(abbreviate_source, axis=1)


# __Test__:

# In[36]:


#Check if the values in the source column of the twitter_archive were changed or not
twitter_archive_clean['source'].value_counts()


# #### Clean: Quality 5

# __Define__: Drop the rating_denominator and change the rating_numerator column to rating. Identify the outliers with rating and eliminate them.
# 
# __Code:__

# In[37]:


twitter_archive_clean.drop(['rating_denominator'], axis=1, inplace=True)

twitter_archive_clean.rename(columns={'rating_numerator': 'rating'}, inplace=True)


# In[38]:


twitter_archive_clean['rating'] = twitter_archive_clean['rating'].astype(float)


# In[39]:


twitter_archive_clean['rating'].sort_values(ascending=False).head(10)


# In[40]:


#check how many ratings are above 14
np.count_nonzero(twitter_archive_clean['rating'] > 14)


# In[41]:


#cheack what type of rating values are above 14
twitter_archive_clean[twitter_archive_clean['rating'] > 14]['rating']


# In[42]:


#eliminate the rows with rating above 14

twitter_archive_clean = twitter_archive_clean[twitter_archive_clean['rating'] <= 14]


# __Test__: 
# 
# 1) The column "rating_denominator" should not be present.
# 
# 2) The column name for "rating_numerator" should be changed to "rating".
# 
# 3) The data type of the rating should be float.
# 
# 4) The number of entries for twitter_arhive_clean should decrease by 26 after eliminating the ratings above 14. 
#    

# In[43]:


twitter_archive_clean.info()


# #### Clean: Quality 6

# __Define:__ Change non-dog names to null in the twitter_archive_clean dataframe. Also, capitalize the first letter of the name.
# 
# __Code:__

# In[44]:


twitter_archive_clean['name'].value_counts()


# In[45]:


#list of non dog names present in the twitter_archive_clean
non_dog_names = ['O', 'a', 'a', 'about', 'above', 'after', 'again', 
             'against', 'all', 'all', 'am', 'an', 'an', 'and', 
             'any', 'are', 'as', 'at', 'at', 'be', 'because', 
             'been', 'before', 'being', 'below', 'between', 
             'both', 'but', 'by', 'by', 'can', 'did', 'do', 
             'does', 'doing', 'don', 'down', 'during', 'each', 
             'few', 'for', 'from', 'further', 'had', 'has', 
             'have', 'having', 'he', 'her', 'here', 'hers', 
             'herself', 'him', 'himself', 'his', 'how', 'i', 
             'if', 'in', 'into', 'is', 'it', 'its', 'itself', 
             'just', 'just', 'life', 'light', 'me', 'more', 
             'most', 'my', 'my', 'myself', 'no', 'nor', 'not', 
             'not', 'now', 'of', 'off', 'old', 'on', 'once', 
             'only', 'or', 'other', 'our', 'ours', 'ourselves', 
             'out', 'over', 'own', 'quite', 's', 'same', 'she', 
             'should', 'so', 'some', 'space', 'such', 'such', 
             't', 'than', 'that', 'the', 'the', 'their', 'theirs', 
             'them', 'themselves', 'then', 'there', 'these', 'they', 
             'this', 'this', 'those', 'through', 'to', 'too', 'under', 
             'until', 'up', 'very', 'very', 'was', 'we', 'were', 
             'what', 'when', 'where', 'which', 'while', 'who', 
             'whom', 'why', 'will', 'with', 'you', 'your', 'yours', 
             'yourself', 'yourselves',"None"]


# In[46]:


#replace these values with nan
twitter_archive_clean['name'] = twitter_archive_clean['name'].replace(non_dog_names,np.nan)


# In[47]:


#lowercase all the dog names
twitter_archive_clean['name'] = twitter_archive_clean['name'].str.lower()


# In[48]:


#capitalize the first letter of the name
twitter_archive_clean['name'] = twitter_archive_clean['name'].str.title()


# __Test__:
# 
# 1) Check if the formatting of all the dog names is consistent
# 2) Spot check any unusal dog names

# In[49]:


twitter_archive_clean['name'].value_counts()


# #### Clean: Quality 7

# __Define__: 
# 
# 1) Drop rows, which contains p1_dog,p2_dog, and p3_dog as false 
# 
# 2) Convert p1,p1_conf,p1_dog to more descriptive column names
# 
# 3) Drop other unnecessary columnns 
# 
# 4) Capitalize the breed name in p1
# 
# 5) Change breed to categorical type
# 
# __Code__:

# In[50]:


image_predictions_clean = image_predictions.copy()


# In[51]:


#Remove Rows, where p1_dog, p2_dog and p3_dog are false
image_predictions_clean = image_predictions_clean[((image_predictions_clean['p1_dog'] == True) & 
                                                  (image_predictions_clean['p2_dog'] == True) &
                                                 (image_predictions_clean['p3_dog'] == True))]


# In[52]:


#Convert p1,p1_conf,p1_dog to more descriptive column names
image_predictions_clean.rename(columns={"img_num":"#_of_images", "p1": "Dog_Breed_predicted",  
                                        "p1_conf":"prediction_confidence", "p1_dog":"Dog_or_Not"}, inplace=True)


# In[53]:


#Drop unnecessary columns from image_predictions dataframe
image_predictions_clean.drop(['p2', 'p2_conf', 'p2_dog', 'p3', 
                  'p3_conf', 'p3_dog'], inplace = True, axis = 1)


# In[54]:


#Capitalize the first letter of the breed name
image_predictions_clean["Dog_Breed_predicted"] = image_predictions_clean["Dog_Breed_predicted"].str.lower()
image_predictions_clean["Dog_Breed_predicted"] = image_predictions_clean["Dog_Breed_predicted"].str.title()


# In[55]:


#Change the type of breed to category type
image_predictions_clean["Dog_Breed_predicted"] = image_predictions_clean["Dog_Breed_predicted"].astype('category')


# In[56]:


#Replace the underscore with the space
image_predictions_clean["Dog_Breed_predicted"] = image_predictions_clean["Dog_Breed_predicted"].str.replace("_"," ")


# __Test__:

# In[57]:


#Check the Breed Name formatting
image_predictions_clean["Dog_Breed_predicted"].value_counts().head(5)


# In[58]:


#Check the remaining columns in the image_predictions_clean dataframe
image_predictions_clean.info()


# In[59]:


#Check if all the rows, which do not contain tweets about the dog are removed
image_predictions_clean['Dog_or_Not'].value_counts()


# #### Clean: Tidiness 1

# __Define:__ Create a column, "dog stage" by extracting values from these four columns (doggo, floofer, pupper, puppo) in the twitter_archive_clean dataframe. If none valules are present in all three columns, then it should be represented as empty entries. In the end, eliminate these four columns (doggo, floofer, pupper, puppo). Also, change the dog_stage to category
# 
# __Code:__

# In[60]:


twitter_archive_clean['doggo'] = twitter_archive_clean['doggo'].replace(np.nan,"None")
twitter_archive_clean['floofer'] = twitter_archive_clean['floofer'].replace(np.nan,"None")
twitter_archive_clean['pupper'] = twitter_archive_clean['pupper'].replace(np.nan,"None")
twitter_archive_clean['puppo'] = twitter_archive_clean['puppo'].replace(np.nan,"None")


# In[61]:


#Replacing None values with empty entries for doggo,floofer, pupper and puppo columns of the twitter_archive_clean dataframe
twitter_archive_clean['doggo'] = twitter_archive_clean['doggo'].replace("None",'')
twitter_archive_clean['floofer'] = twitter_archive_clean['floofer'].replace("None",'')
twitter_archive_clean['pupper'] = twitter_archive_clean['pupper'].replace("None",'')
twitter_archive_clean['puppo'] = twitter_archive_clean['puppo'].replace("None",'')


# In[62]:


#Creating "dog stage" column by extracting values from four columns (doggo, floofer, pupper, puppo) 
#of twitter_archive_clean dataframe

twitter_archive_clean['dog_stage'] = twitter_archive_clean['doggo'] + twitter_archive_clean['floofer'] + twitter_archive_clean['pupper'] + twitter_archive_clean['puppo'] 


# In[63]:


#Check the values present in the dog_stage column
twitter_archive_clean['dog_stage'].value_counts()


# In[64]:


#Format the various dog_stages in a proper format

twitter_archive_clean.loc[twitter_archive_clean['dog_stage'] == "pupper","dog_stage"] = "Pupper"
twitter_archive_clean.loc[twitter_archive_clean['dog_stage'] == "doggo","dog_stage"] = "Doggo"
twitter_archive_clean.loc[twitter_archive_clean['dog_stage'] == "puppo","dog_stage"] = "Puppo"
twitter_archive_clean.loc[twitter_archive_clean['dog_stage'] == "doggopuppo","dog_stage"] = "Doggo, Puppo"
twitter_archive_clean.loc[twitter_archive_clean['dog_stage'] == "doggopupper","dog_stage"] = "Doggo, Pupper"
twitter_archive_clean.loc[twitter_archive_clean['dog_stage'] == "doggofloofer","dog_stage"] = "Doggo, Floffer"


# In[65]:


#Check if the correct values are present in the dog_stage column before removed other four columns
twitter_archive_clean[['dog_stage','doggo','floofer','pupper','puppo']].sample(10)


# In[66]:


#Drop doggo, floofer, pupper, puppo columns from the twitter_archive_clean dataframe
twitter_archive_clean.drop(['doggo','floofer','pupper','puppo'], axis = 1, inplace=True)


# In[67]:


#Convert the dog_stage to category
twitter_archive_clean['dog_stage']= twitter_archive_clean['dog_stage'].astype('category')


# __Test__:

# In[68]:


#Check which values are present in the dog stage column
twitter_archive_clean['dog_stage'].value_counts()


# In[69]:


#Check if the four dog stage columns were dropped or not
#Check if the data type for dog_stage was changed to category
twitter_archive_clean.info()


# #### Clean: Tidiness 2

# __Define:__ Join the three dataframes (twitter_archive, image_predictions, and tweet_info) into one master dataset via inner join based on tweet_id
# 
# __Code:__

# In[70]:


twitter_archive_master = pd.merge(twitter_archive_clean, tweet_info_clean, on='tweet_id', how = 'inner')

twitter_archive_master = pd.merge(twitter_archive_master, image_predictions_clean, on='tweet_id')
                                  


# __Test__: 

# In[71]:


#Check if twitter_archive_master contains all the columns present in the final clean copies of these three dataframes 
#(twitter_archive_clean, image_predictions_clean, and tweet_info_clean)

twitter_archive_master.info()


# #### Clean: Quality 8

# __Define:__ Drop the columns from the twitter_archive_master dataframe, which won't be required for the analysis. Also, reindex all the rows of the twitter_archive_master dataframe
# 
# __Code:__

# In[72]:


twitter_archive_master.drop(['expanded_urls','#_of_images','prediction_confidence','Dog_or_Not'],axis=1, inplace=True)


# In[73]:


#reindex all the rows of the twitter_achive_master dataframe
twitter_archive_master.reindex(columns=['tweet_id','timestamp','source','text','rating','name','Day','Month','Year','Time',
 'weekday','dog_stage','retweet_count','favorite_count','jpg_url','Dog_Breed_predicted']).tail(3)


# __Test:__

# In[74]:


twitter_archive_master.info()


# In[75]:


#Check the index of first 5 rows
twitter_archive_master.head(5)


# #### Clean: Quality 9

# __Define:__ Convert datatype of tweet_id to string of twitter_archive_master
# 
# __Code:__

# In[76]:


twitter_archive_master['tweet_id'] = twitter_archive_master['tweet_id'].astype(str)


# __Test__: 

# In[77]:


#Check the datatype of twitter_archive_master
type(twitter_archive_master['tweet_id'][1])


# ## Store

# In[78]:


#store the clean dataframe in a csv file

twitter_archive_master.to_csv('twitter_archive_master.csv', index=False)


# ## Analysis & Visualizations

# We know three charactericts about our subject, dog, which are as following: dog name, dog stage, and dog breed.
# 
# The analysis of these three charactericts are based on rating, retweets, and favorites.
# 
# Following questions will be investigated in this section.
# 
# 1) Which dog breeds are most popular?
# 
# 2) Which dog breeds have been most retweeted?
# 
# 3) What is the most common dog stage?
# 
# 4) What is the most popular dog names?
# 
# 5) What is the most popular method for posting tweets?
# 
# 6) How the number of tweets have changed over period of time?
# 

# In[79]:


plt.style.use('bmh')

sns.countplot(data=twitter_archive_clean, y='source')
plt.title('Tweet Source Distribution')
plt.xlabel('Number of Tweets')
plt.ylabel('Source')
plt.savefig('tweet-source.png')


# Twitter App is manily used. Majority of the twitter users use iPhone twitter App. While the small percentage of the population uses Twitter Web Client, Vine and TweetDeck.

# In[80]:


#Top favorite dogs

twitter_archive_master.sort_values(by=['favorite_count'],ascending = False).head(1)


# The most favorite dog is Lakeland Terrior.

# In[102]:


#Top retweet dog

twitter_archive_master.sort_values(by=['retweet_count'],ascending = False).head(1)


# The most retweeted dog is Eskimo Dog.

# In[82]:


#Visualize relative share of dog_stages
dog_stage_counts = twitter_archive_master['dog_stage'].value_counts()[1:5]


# In[83]:


dog_stage_counts


# In[84]:


labels = []
denominator = dog_stage_counts.sum()

for index, count in enumerate(dog_stage_counts):
    label_first_part = dog_stage_counts.index.values[index]
    label_second_part = (count / denominator) * 100
    label_second_part = round(label_second_part, 2)
    label_second_part = str(label_second_part) + '%'
    label = label_first_part + ' ' + label_second_part
    labels.append(label)


# In[85]:


plt.figure(figsize=(12, 8))
plt.pie(dog_stage_counts, labels = labels, explode = (0.1,0.1,0.1,0.1), 
        shadow = True, startangle = 90)
plt.title('Share of Dog Stages')
plt.savefig('Share od Dog Stages')


# The most owned dog stage is Pupper, followed by Doggo, Puppo and Floofer.

# In[86]:


twitter_archive_master.groupby('dog_stage')['rating'].mean()


# Eventhough, Pupper is the most owned dog, it has lowest rating. Doggo has the highest rating.

# In[87]:


#Visaulize the top 10 dog breeds
top_breeds = twitter_archive_master['Dog_Breed_predicted'].value_counts()[0:10].sort_values(axis=0,ascending=False)

top_breeds.plot(kind='barh',color=['steelblue'])

plt.title('Top 10 Breeds')

plt.xlabel('Count')

plt.ylabel('Breed')

plt.savefig('top_10_breeds.png')


# The most common breed type is Golden Retriver, followed by Pembroke and Labrador Retriever.

# In[88]:


#Visualize the top 10 dog names
top_breeds = twitter_archive_master['name'].value_counts()[0:10].sort_values(axis=0,ascending=False)

top_breeds.plot(kind='barh',color=['steelblue'])

plt.title('Top 10 Names')

plt.xlabel('Count')

plt.ylabel('Names')

plt.savefig('top_10_names.png')


# The most common dog name is Copper, follower by Oliver and Charlie.

# In[89]:


#Retweet and Favorites Scatter Plot

twitter_archive_master.plot(kind='scatter',x='favorite_count',y='retweet_count',alpha=0.5)

plt.xlabel('Favorites Count')
plt.ylabel('Retweets Count')
plt.title('Retweets and Favorites Scatter plot')


# The above graph shows strong correlation betwwen the Retweet and Favorite, options available on twitter. As the tweet gets more re-tweeted, it has higher chance of getting "favorite" click. Therefore, this relationship makes sense. Also, from the below chart, one can see that correlation coefficient between favorite and retweet is 0.93, which reinforces strong relationship between among them. 

# In[90]:


twitter_archive_master[['rating','favorite_count','retweet_count']].corr(method='pearson')


# From the above correlation coefficient values, one can see that dog rating has weak relationship with the favorite and retweet options.  

# In[91]:


twitter_archive_master[['favorite_count','retweet_count']].plot(style='.',ylim=[0,50000])
plt.title('Favorites, Retweets over Time')
plt.xlabel('Time')
plt.xticks([],[])
plt.ylabel('Count')
plt.legend(ncol=1,loc='upper right')
plt.savefig('retweets-favorites-time.png')


# Favorites are more popular than retweets. Both number of favorite and retweet counts are decreasing with Time.

# In[92]:


twitter_archive_master['rating'].plot(style='o',alpha=0.5)
plt.title('Rating over Time')
plt.xlabel('Date')
plt.xticks([],[])
plt.ylabel('Rating')
plt.savefig('rating_over_time.png')


# One can see, from the above graph that Rating has overall decreased over time.

# In[107]:


tweets_weekday = twitter_archive_master['weekday'].value_counts()
tweets_weekday.plot(kind='barh')
plt.ylabel('weekday')
plt.xlabel('number of tweets')
plt.savefig('tweets_weekdays.png')


# In[108]:


tweets_month = twitter_archive_master['Month'].value_counts()
tweets_month.plot(kind='barh')
plt.ylabel('month')
plt.xlabel('number of tweets')
plt.savefig('tweets_month.png')


# The most number of tweets occur in the month of december and on Monday.
