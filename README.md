# Cyclistic Bike-Share Analysis - README

## 📊 Project Overview
This analysis examines usage patterns between casual riders and annual members in Cyclistic's bike-share program. The dataset includes **5,400,488 rides** during January - December 2025, with detailed breakdowns by user type, time, season, and bike preference.

---

## 🔑 KEY INSIGHTS & SURPRISES

### 📅 DAY PATTERNS:
- **Casual riders** use bikes more on weekends: **36.6%** of their rides occur on Saturday-Sunday (vs. **23.3%** for members)
- **Members** show consistent weekday usage, peaking on **Thursdays**
- Casual riders' most popular day: **Saturday** (20.6% of their total rides)
- Members' most popular day: **Thursday** (16.2% of their total rides)

### ⏱️ DURATION DIFFERENCES:
- **Casual rides are 7.7 minutes longer on average** (19.9 min vs. 12.2 min)
- Casual riders show longer median durations: **11.9 min** vs. **8.73 min** for members
- Weekend casual rides are particularly long: **23.1 min** on Sundays

### 🕐 PEAK USAGE TIMES:
- **Both user types peak at 17:00** (5 PM)
- Members show clear **commute patterns**: high usage at 7-9 AM and 4-6 PM
- Casual riders have more **distributed afternoon usage** (10 AM-7 PM)
- Member morning commute is stronger: **7:00-8:00** shows 196K-252K rides

### 🌤️ SEASONAL PATTERNS:
- **Casual ridership triples in summer** (June-August vs. winter months)
- Members show **more consistent year-round usage** with moderate summer increase
- Casual usage peaks in **August** (323,560 rides)
- Member usage peaks in **August** (443,173 rides)

### 🚲 BIKE PREFERENCES:
- **Electric bikes dominate** for both groups: 64% of member rides, 65% of casual rides
- **Casual riders** use **classic bikes for longer durations** (29.26 min avg)
- **Members** show shorter, more consistent durations across both bike types

---

## 💡 ANSWERING BUSINESS QUESTIONS

### HOW CASUAL AND MEMBER RIDERS DIFFER:

#### 1. USAGE PURPOSE (Inferred):
- **Members: Primarily commute/work-related**
  - Peak at traditional commute hours (7-9 AM, 4-6 PM)
  - Consistent weekday usage (Monday-Friday)
  - Shorter, predictable ride lengths (12.2 min average)
  - Higher usage on Thursdays (peak workday)

- **Casual: Primarily leisure/recreational**
  - Higher weekend usage (36.6% on weekends)
  - Longer ride durations (19.9 min average)
  - More seasonal variation (3x summer increase)
  - Afternoon-focused usage patterns

#### 2. CONVERSION OPPORTUNITIES:
- **Target weekend casual riders** with 'Weekend Warrior' membership
- **Offer summer promotion packages** to capture seasonal users
- **Create electric bike-focused membership tier** (popular with casual riders)
- **Develop off-peak hour incentives** for members to balance system usage

#### 3. MARKETING STRATEGY IMPLICATIONS:
- **Digital ads should target weekends and summer months**
- **Highlight electric bike access** in membership promotions
- **Create 'commute calculator'** to show membership savings vs. casual pricing
- **Develop app features for leisure route discovery** to attract casual users

---

## 📊 QUANTITATIVE FINDINGS:
1. **Scale**: 5,400,488 total rides analyzed
2. **User Split**: 35.5% casual (1,915,996 rides) vs. 64.5% member (3,484,492 rides)
3. **Duration Gap**: Casual rides 7.7 minutes longer on average (19.9 vs. 12.2 min)
4. **Peak Consistency**: Both groups peak at 17:00, but members show stronger morning commute
5. **Seasonality Impact**: Casual usage increases ~300% summer vs. winter; members increase ~200%

---

## 🎯 QUALITATIVE INSIGHTS:
1. **Members = Commuters** (predictable, short, weekday-focused, consistent year-round)
2. **Casual = Leisure** (flexible, long, weekend/seasonal, afternoon-focused)
3. **Electric bike preference** is strong across both groups, but casual riders use classic bikes longer
4. **Clear conversion opportunity**: Weekend leisure riders represent prime targets for membership

---

## 💼 BUSINESS RECOMMENDATIONS:

### 1. **"Weekend Pass" Membership Tier**
- Target: Casual weekend riders (36.6% of their usage)
- Pricing: Between daily and annual membership
- Benefits: Unlimited weekend rides, discounted weekdays

### 2. **Summer Marketing Campaign**
- Launch: Early May to capture seasonal increase
- Channels: Tourist areas, hotels, summer event partnerships
- Offer: "Summer Starter" membership with first month discount

### 3. **Electric Bike Membership Benefits**
- Highlight in marketing: Casual riders prefer electric bikes (65% of their rides)
- Create "E-Bike Priority" membership tier
- Develop safety/tutorial content for new e-bike users

### 4. **Commute Cost Savings Campaign**
- Develop calculator showing annual savings vs. casual pricing
- Partner with local employers for corporate membership programs
- Target marketing near business districts during commute hours

### 5. **Data-Driven Feature Development**
- App feature: "Weekend Adventure Routes" for casual users
- Notification system: Alert casual riders about membership benefits after 3+ rides/week
- Personalized offers: Target frequent casual users with conversion incentives

---

## 📈 SUCCESS METRICS FOR IMPLEMENTATION:
- Increase casual-to-member conversion rate by 15% in 6 months
- Boost weekend member rides by 20%
- Increase summer membership sales by 25%
- Improve electric bike utilization balance across user types

---

Data Source: Divvy Bike Share Data (public dataset)
Disclaimer: This is a capstone project for the Google Data Analytics Certificate

*Analysis based on Cyclistic bike-share data. All insights derived from the provided dataset spanning multiple months of ridership patterns.*
