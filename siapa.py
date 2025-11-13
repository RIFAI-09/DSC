import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import ccf

#1 load data dulu
sales = pd.read.csv('sales.csv')
products = pd.read.csv('products.csv')
marketing = pd.read.csv('marketing.csv')
reviews = pd.read.csv('reviews.csv')

#2 data prepraration
#convert kolom tanggal
sales['date'] = pd.to_datetime(sales['date'], format='%Y-%m-%d')
products['launch_date'] = pd.to_datetime(products['launch_date'], format='%Y-%m-%d')

#merge data
sales_full = sales.merge(products, on='product_id', how='left')

#aggregate daily sales per product
daily_sales = (sales_full.groupby(['date', 'product_id'])
               .agg({'units_sold':'sum', 'revenue':'sum', 'avg_price':'mean'})
               .reset_index()
)

#add launch info
daily_sales = daily_sales.merge(products[['product_id','brand','launch_date']], on='product_id', how='left')
daily_sales['days_since_launch'] = (daily_sales['date'] - daily_sales['launch_date']).dt.days

#3 visual check, sales overtime
plt.figure(figsize=(12,6))
for pid in daily_sales['product_id'].unique():
    temp = daily_sales[daily_sales['product_id']==pid]
    plt.plot(temp['date'], temp['revenue'], label=pid, alpha=0.7)
plt.title("Revenue Trend per Product")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()

#4 cannibalization logic
#hipotesis: Jika produk baru launching, maka produk lamaa akan mengalami penurunan sales

#step1
brands = products['brand'].unique()
cannibal_effect = []

for brand in brands:
    brand_products = products[products['brand']==brand].sort_values('launch_date')
    brand_sales = daily_sales[daily_sales['product_id']==brand]
    
    for i in range(1, len(brand_products)):
        new_product = brand_products.iloc[i]
        old_product = brand_products.iloc[i-1]
        
        #extract overlap period
        launch_date = new_product['launch_date']
        window_start = launch_date - pd.Timedelta(days=180)
        window_end = launch_date + pd.Timedelta(days=180)
        
        old_sales = brand_sales[(brand_sales['product_id']==old_product['product_id']) &
                                (brand_sales['date'].between(window_start, window_end))]
        new_sales = brand_sales[(brand_sales['product_id']==new_product['product_id']) &
                                (brand_sales['date'].between(window_start, window_end))]
        
        #aggregate weekly revenue
        old_weekly = old_sales.resample('W-Mon', on='date')['revenue'].sum()
        new_weekly = new_sales.resample('W-Mon', on='date')['revenue'].sum()
        
        #align index
        alligned = pd.DataFrame({'old': old_weekly, 'new': new_weekly}).fillna()
        
        #compute cross-correlation
        corr = np.corrcoef(alligned['old'], alligned['new'])[0,1]
        
        cannibal_effect.append({
            'brand': brand,
            'old_product': old_product['product_name'],
            'new_product': new_product['product_name'],
            'correlation': corr
        })
        
cannibal_df = pd.DataFrame(cannibal_effect)
print("\n=== Cannibalization Analysis Results ===")
print(cannibal_df.sort_values('correlation').head(10))

#5 visualize correlation
plt.figure(figsize=(8,5))
sns.barplot(data=cannibal_df.sort_values('correlation'), x='brand', y='correlation', hue='brand', dodge=False)
plt.axvline(0, color='red', linestyle='--')
plt.title("Cannibalization Correlation per Brand")
plt.xlabel("Correlation Coefficient (negative = potential cannibalization)")
plt.tight_layout()
plt.show()

#6 insight (sentiment & marketing support)
#sentiment
if 'rating' in reviews.columns:
    sentiment_summary = reviews.groupby('product_id')['rating'].mean().reset_index()
    cannibal_df = cannibal_df.merge(products[['product_id', 'product_name', 'brand']],
                                    left_on='new_product', right_on='product_name', how='left')
    cannibal_df = cannibal_df.merge(sentiment_summary, on='product_id', how='left')
    print("\n=== Average Sentiment for new Products in Cannibalization Pairs ===")
    print(cannibal_df[['brand', 'new_product', 'correlation', 'rating']])
    
#7 marketing spend effect
marketing_summary = marketing.groupby('product_id')['spend'].sum().reset_index()
cannibal_df = cannibal_df.merge(products[['product_id','product_name']], 
                                left_on='new_product', right_on='product_name', how='left')
cannibal_df = cannibal_df.merge(marketing_summary, on='product_id', how='left')

plt.figure(figsize=(7,5))
sns.scatterplot(data=cannibal_df, x='spend', y='corr', hue='brand')
plt.title("Marketing Spend vs Cannibalization Correlation")
plt.xlabel("Total Marketing Spend (IDR)")
plt.ylabel("Correlation (negative = stronger cannibalization)")
plt.tight_layout()
plt.show()