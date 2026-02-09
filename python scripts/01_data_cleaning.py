#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install and load relevant libraries
import pandas as pd
import numpy as np
from datetime import datetime
import warnings


# In[2]:


# Upload the Cyclistic datasets (csv files) from the past 12 months
jan_trips = pd.read_csv("2025_01_cyclistic.csv")
feb_trips = pd.read_csv("2025_02_cyclistic.csv")
mar_trips = pd.read_csv("2025_03_cyclistic.csv")
apr_trips = pd.read_csv("2025_04_cyclistic.csv")
may_trips = pd.read_csv("2025_05_cyclistic.csv")
june_trips = pd.read_csv("2025_06_cyclistic.csv")
july_trips = pd.read_csv("2025_07_cyclistic.csv")
aug_trips = pd.read_csv("2025_08_cyclistic.csv")
sep_trips = pd.read_csv("2025_09_cyclistic.csv")
oct_trips = pd.read_csv("2025_10_cyclistic.csv")
nov_trips = pd.read_csv("2025_11_cyclistic.csv")
dec_trips = pd.read_csv("2025_12_cyclistic.csv")


# In[3]:


# Check if all DataFrames have identical columns
print("Checking column consistency across all months...")

# Get column sets for each month
datasets = {
    'Jan': jan_trips, 'Feb': feb_trips, 'Mar': mar_trips,
    'Apr': apr_trips, 'May': may_trips, 'Jun': june_trips,
    'Jul': july_trips, 'Aug': aug_trips, 'Sep': sep_trips,
    'Oct': oct_trips, 'Nov': nov_trips, 'Dec': dec_trips
}

# Check column names match
base_columns = set(jan_trips.columns)
all_match = True

for month, df in datasets.items():
    if set(df.columns) != base_columns:
        print(f"⚠️  {month} has different columns!")
        print(f"   Extra: {set(df.columns) - base_columns}")
        print(f"   Missing: {base_columns - set(df.columns)}")
        all_match = False

if all_match:
    print("✅ All 12 months have identical column structure")
    print(f"Columns ({len(base_columns)}): {list(base_columns)}")


# In[4]:


# Stack individual month's data frames into one big data frame
all_trips = pd.concat([
    jan_trips, feb_trips, mar_trips, apr_trips, may_trips, june_trips,
    july_trips, aug_trips, sep_trips, oct_trips, nov_trips, dec_trips
], ignore_index=True)

print(f"✅ Combined all 12 months: {len(all_trips):,} total rows")
print(f"Memory usage: {all_trips.memory_usage(deep=True).sum() / 1024**2:.1f} MB")


# In[5]:


# Save combined raw data before cleaning
raw_data_path = '2025_cyclistic_combined_raw.csv'
all_trips.to_csv(raw_data_path, index=False)
print(f"Saved combined raw data to: {raw_data_path}")


# In[6]:


# 1. Load raw data
print("\n1. Loading saved raw data...")
df = pd.read_csv('2025_cyclistic_combined_raw.csv')
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"Date range in raw data: {df['started_at'].iloc[0]} to {df['started_at'].iloc[-1]}")


# In[7]:


# 2. Convert datetime columns with error handling
print("\n2. Converting datetime columns...")
df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')

# Check for failed conversions
failed_start = df['started_at'].isna().sum()
failed_end = df['ended_at'].isna().sum()

if failed_start > 0 or failed_end > 0:
    print(f"Warning: {failed_start} started_at and {failed_end} ended_at conversions failed")
    print("These rows will be removed later.")
else:
    print("All datetime conversions successful")


# In[8]:


# 3. Standardize member_casual column
print("\n3. Standardizing member_casual column...")

print("Before standardization:")
print(df['member_casual'].value_counts(dropna=False))

# Standardize: strip whitespace, lowercase
df['member_casual'] = df['member_casual'].astype(str).str.strip().str.lower()

print("\nAfter standardization:")
print(df['member_casual'].value_counts(dropna=False))


# In[9]:


# 4. Convert ride_length to minutes
print("\n4. Converting ride_length to minutes...")

# Check current format of ride_length
print(f"ride_length sample values:")
print(df['ride_length'].head(3).tolist())

# Function to convert hh:mm:ss to minutes
def convert_ride_length(time_str):
    """Convert hh:mm:ss string to total minutes (float)"""
    try:
        if pd.isna(time_str):
            return None
        
        # Handle different formats
        if isinstance(time_str, str):
            # Format: "hh:mm:ss" or "mm:ss"
            parts = time_str.split(':')
            
            if len(parts) == 3:  # hh:mm:ss
                hours, minutes, seconds = map(int, parts)
                return hours * 60 + minutes + seconds / 60
            elif len(parts) == 2:  # mm:ss
                minutes, seconds = map(int, parts)
                return minutes + seconds / 60
            else:
                return None
        elif isinstance(time_str, (int, float)):
            # If already numeric, assume it's seconds
            return time_str / 60
        else:
            return None
    except:
        return None

# Apply conversion
df['ride_length_min'] = df['ride_length'].apply(convert_ride_length)

print(f"\nConversion check (first 5 rows):")
print(df[['ride_length', 'ride_length_min']].head())

print(f"\nRide length statistics (minutes):")
print(f"  Min: {df['ride_length_min'].min():.2f}")
print(f"  Max: {df['ride_length_min'].max():.2f}")
print(f"  Mean: {df['ride_length_min'].mean():.2f}")
print(f"  Missing: {df['ride_length_min'].isna().sum()}")


# In[10]:


# 5. Convert day_of_week to day names
print("\n5. Converting day_of_week numbers to day names...")

# Check current day_of_week values
print(f"Current day_of_week values (1-7, Sunday=1):")
print(df['day_of_week'].value_counts().sort_index())

# Map numeric to day names
day_map = {
    1: 'Sunday',
    2: 'Monday',
    3: 'Tuesday',
    4: 'Wednesday',
    5: 'Thursday',
    6: 'Friday',
    7: 'Saturday'
}

# Create day_name column
df['day_name'] = df['day_of_week'].map(day_map)

print(f"\nDay distribution (from numeric column):")
print(df['day_name'].value_counts().sort_index())


# In[11]:


# 6. Add additional time features
print("\n6. Adding additional time features...")

# Month and year
df['month'] = df['started_at'].dt.month_name()
df['year'] = df['started_at'].dt.year

# Hour of day
df['hour'] = df['started_at'].dt.hour
df['hour_category'] = pd.cut(df['hour'], 
                             bins=[0, 6, 10, 15, 19, 24],
                             labels=['Night (0-6)', 'Morning (7-10)', 'Midday (11-15)', 
                                     'Evening (16-19)', 'Night (20-23)'],
                             right=False)

# Date (without time)
df['date'] = df['started_at'].dt.date

# Season (based on month)
def get_season(month_name):
    """Convert month_name to season with datetime - FIXED"""
    # Convert to string first
    month_str = str(month_name)
    try:
        month_num = datetime.strptime(month_str, "%B").month
        if month_num in [12, 1, 2]:
            return 'Winter'
        elif month_num in [3, 4, 5]:
            return 'Spring'
        elif month_num in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    except:
        return 'Unknown'

df['season'] = df['month'].apply(get_season)

print("Added: month, year, hour, hour_category, date, season")


# In[12]:


# 7. Filter invalid data
print("\n7. Filtering invalid data...")

original_count = len(df)
print(f"Original row count: {original_count:,}")

# Define filtering conditions
conditions = [
    df['ride_length_min'].notna(),                    # Has valid ride length
    df['ride_length_min'] >= 1,                       # At least 1 minute
    df['ride_length_min'] <= 1440,                    # Max 24 hours (1440 minutes)
    df['member_casual'].isin(['member', 'casual']),   # Valid user type
    df['started_at'].notna(),                         # Has start time
    df['ended_at'].notna(),                           # Has end time
    df['started_at'] < df['ended_at'],                # Logical time order
    df['rideable_type'].notna()                       # Has bike type
]

# Apply all conditions
clean_df = df.copy()
for i, condition in enumerate(conditions, 1):
    before = len(clean_df)
    clean_df = clean_df[condition].copy()
    removed = before - len(clean_df)
    if removed > 0:
        print(f"  Condition {i}: Removed {removed:,} rows")

# Final count
final_count = len(clean_df)
removed_total = original_count - final_count
removal_percent = (removed_total / original_count) * 100

print(f"\nFILTERING SUMMARY:")
print(f"  Original rows: {original_count:,}")
print(f"  Removed rows: {removed_total:,} ({removal_percent:.2f}%)")
print(f"  Final clean rows: {final_count:,}")


# In[13]:


# 8. Data Quality Report
print("\n8. Generating data quality report...")

print("\nDATA QUALITY CHECKLIST:")
print("-" * 40)

# 1. User type distribution
print("1. User Type Distribution:")
user_counts = clean_df['member_casual'].value_counts()
for user_type, count in user_counts.items():
    percent = (count / len(clean_df)) * 100
    print(f"   {user_type.title()}: {count:,} rides ({percent:.1f}%)")

# 2. Date range
print(f"\n2. Date Range:")
print(f"   Start: {clean_df['started_at'].min()}")
print(f"   End: {clean_df['started_at'].max()}")
print(f"   Total days: {(clean_df['started_at'].max() - clean_df['started_at'].min()).days}")

# 3. Ride length statistics
print(f"\n3. Ride Length Statistics (minutes):")
for user_type in ['member', 'casual']:
    user_data = clean_df[clean_df['member_casual'] == user_type]
    print(f"   {user_type.title()}:")
    print(f"     Avg: {user_data['ride_length_min'].mean():.1f} min")
    print(f"     Median: {user_data['ride_length_min'].median():.1f} min")
    print(f"     Max: {user_data['ride_length_min'].max():.1f} min")

# 4. Missing values
print(f"\n4. Missing Values in Clean Data:")
missing_clean = clean_df.isnull().sum()
missing_clean = missing_clean[missing_clean > 0]
if len(missing_clean) == 0:
    print("   No missing values in critical columns")
else:
    for col, count in missing_clean.items():
        print(f"   {col}: {count} missing ({count/len(clean_df)*100:.2f}%)")


# In[14]:


# 9. Save cleaned data
print("\n9. Saving cleaned data...")

# Save full cleaned dataset
clean_df.to_csv('2025_cyclistic_cleaned_full.csv', index=False)
print(f"Saved full cleaned data: 2025_cyclistic_cleaned_full.csv")

# Save analysis-ready summary (smaller, faster for analysis)
analysis_cols = [
    'ride_id', 'rideable_type', 'started_at', 'ended_at',
    'member_casual', 'ride_length_min', 'day_name', 
    'month', 'year', 'hour', 'hour_category', 'season'
]

clean_df[analysis_cols].to_csv('2025_cyclistic_analysis_ready.csv', index=False)
print(f"Saved analysis-ready data: 2025_cyclistic_analysis_ready.csv")


# In[15]:


# 10. Create quick summary for Excel
print("\n10. Creating summary for Excel visualization...")

# Create summary tables for easy Excel import
with pd.ExcelWriter('2025_cyclistic_summaries.xlsx') as writer:
    
    # 1. Monthly usage by user type
    monthly_summary = pd.crosstab(
        clean_df['month'], 
        clean_df['member_casual'],
        values=clean_df['ride_id'],
        aggfunc='count',
        margins=True,
        margins_name='Total'
    )
    monthly_summary.to_excel(writer, sheet_name='Monthly_Usage')
    
    # 2. Day of week patterns
    dow_summary = pd.crosstab(
        clean_df['day_name'], 
        clean_df['member_casual'],
        values=clean_df['ride_length_min'],
        aggfunc='mean'
    ).round(2)
    dow_summary.to_excel(writer, sheet_name='Day_of_Week')
    
    # 3. Hourly patterns
    hourly_summary = pd.crosstab(
        clean_df['hour'], 
        clean_df['member_casual'],
        values=clean_df['ride_id'],
        aggfunc='count'
    )
    hourly_summary.to_excel(writer, sheet_name='Hourly_Usage')
    
    # 4. Bike type preference
    bike_summary = pd.crosstab(
        clean_df['rideable_type'], 
        clean_df['member_casual'],
        normalize='columns'
    ).round(4) * 100
    bike_summary.to_excel(writer, sheet_name='Bike_Preference')
    
    # 5. Basic statistics
    stats_summary = clean_df.groupby('member_casual').agg({
        'ride_id': 'count',
        'ride_length_min': ['mean', 'median', 'std', 'min', 'max']
    }).round(2)
    stats_summary.to_excel(writer, sheet_name='Basic_Stats')

print(f"Created Excel summary: 2025_cyclistic_summaries.xlsx")


# In[16]:


# Final Report
print("\n" + "="*60)
print("🎉 CLEANING PROCESS COMPLETE!")
print("="*60)

print(f"""
📋 FINAL RESULTS:
   • Clean rides: {len(clean_df):,}
   • Casual riders: {user_counts.get('casual', 0):,}
   • Annual members: {user_counts.get('member', 0):,}
   • Data quality: {(len(clean_df)/original_count*100):.1f}% retained

📁 OUTPUT FILES CREATED:
   1. 2025_cyclistic_cleaned_full.csv       (Complete cleaned dataset)
   2. 2025_cyclistic_analysis_ready.csv     (Analysis-optimized dataset)
   3. 2025_cyclistic_summaries.xlsx         (Excel-ready summaries)

📊 NEXT STEPS:
   1. Open '2025_cyclistic_summaries.xlsx' for initial insights
   2. Use cleaned data for in-depth analysis
   3. Create visualizations comparing member vs. casual usage
""")


# In[ ]:




