#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


matches = pd.read_csv("matches.csv",index_col=0)


# In[3]:


matches.head()


# In[4]:


matches.shape


# In[5]:


matches["team"].value_counts()


# In[7]:


matches[matches["team"] == "Liverpool"].sort_values("date")


# In[8]:


matches["round"].value_counts()


# In[9]:


matches.dtypes


# In[10]:


del matches["comp"]


# In[11]:


matches["date"] = pd.to_datetime(matches["date"])


# In[12]:


matches["target"] = (matches["result"] == "W").astype("int")


# In[13]:


matches


# In[14]:


matches["venue_code"] = matches["venue"].astype("category").cat.codes


# In[15]:


matches["opp_code"] = matches["opponent"].astype("category").cat.codes


# In[16]:


matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")


# In[17]:


matches["day_code"] = matches["date"].dt.dayofweek


# In[18]:


matches


# In[19]:


from sklearn.ensemble import RandomForestClassifier


# In[20]:


rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)


# In[21]:


train = matches[matches["date"] < '2022-01-01']


# In[22]:


test = matches[matches["date"] > '2022-01-01']


# In[23]:


predictors = ["venue_code", "opp_code", "hour", "day_code"]


# In[24]:


rf.fit(train[predictors], train["target"])


# In[25]:


preds = rf.predict(test[predictors])


# In[26]:


from sklearn.metrics import accuracy_score


# In[27]:


error = accuracy_score(test["target"], preds)


# In[28]:


error


# In[29]:


combined = pd.DataFrame(dict(actual=test["target"], predicted=preds))


# In[30]:


pd.crosstab(index=combined["actual"], columns=combined["predicted"])


# In[32]:


from sklearn.metrics import precision_score

precision_score(test["target"], preds)


# In[33]:


grouped_matches = matches.groupby("team")


# In[34]:


group = grouped_matches.get_group("Manchester City").sort_values("date")


# In[35]:


def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


# In[37]:


cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

rolling_averages(group, cols, new_cols)


# In[38]:


matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))


# In[39]:


matches_rolling


# In[40]:


matches_rolling = matches_rolling.droplevel('team')


# In[41]:


matches_rolling


# In[42]:


matches_rolling.index = range(matches_rolling.shape[0])


# In[43]:


def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    error = precision_score(test["target"], preds)
    return combined, error


# In[44]:


combined, error = make_predictions(matches_rolling, predictors + new_cols)


# In[45]:


error


# In[46]:


combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)


# In[47]:


combined.head(10)


# In[48]:


class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd", "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"} 
mapping = MissingDict(**map_values)


# In[49]:


combined["new_team"] = combined["team"].map(mapping)


# In[50]:


merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])


# In[51]:


merged


# In[52]:


merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()


# In[ ]:




