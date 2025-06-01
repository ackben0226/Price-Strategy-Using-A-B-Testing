# Price Optimization Using A/B Testing: Data-Driven Insights for Retail Strategy
## Executive Summary
This project evaluates the impact of **three pricing strategies—discounts, price increases, and product bundling—on sales, revenue, and customer behavior** using A/B testing. By analyzing transaction data from a retail business, we determine:

- __Whether a 10% discount increases sales volume and revenue.__
- __If a 10% price increase affects customer demand.__
- __How bundling products influences purchase behavior.__

## Key Findings
✅ __Discount Strategy (10% off):__
- **Increased sales volume by 15%** but only **5% revenue growth** due to lower margins.
- Significant impact in **Electronics (p=0.005)** and **Clothing (p=0.0079)**.
- **Beauty & Books showed minimal revenue change**, suggesting price elasticity varies by category.

✅ __Price Increase (10% higher):__
- **3% drop in sales volume** but **7% revenue increase**, indicating **price-insensitive customers** in some categories.
- **Electronics & Home categories tolerated price hikes better** than others.

✅ **Bundling Strategy:**
- **20% higher average order value** compared to individual sales.
- **30% adoption rate** when bundling complementary products (e.g., Beauty + Home).

## Actionable Insights
**1. Targeted Discounts:**
   - Apply discounts to **high-elasticity categories** (Electronics, Clothing).
   - Avoid deep discounts in **low-elasticity categories** (Beauty, Books).

**2. Strategic Price Increases:**
   - Test small price hikes in **premium categories** (Electronics, Home).
   - Monitor customer retention post-increase.

**3. Promote Bundles:**
   - Bundle **frequently co-purchased items** (e.g., Beauty + Home).
   - Offer **limited-time bundle deals** to boost adoption.




## 1. __Project Overview__
### Objective
<br/> Determine how pricing strategies affect:
- __Sales Volume__
- __Revenue__
- __Customer Behavior__

## __Actions:__ 
- ### __Data Collection and Preparation__
   - Collected sales transaction data, including product categories, pricing, quantity sold, and revenue.
   - Cleaned and preprocessed the data to handle missing values and outliers, ensuring high data quality for analysis.
- ### __Experimental Design__
   - __A/B testing:__ Compared control (**original pricing**) vs. test groups (**discounts, price hikes, bundles**).
   - __Statistical Analysis:__ Used __t-tests & Mann-Whitney U tests__ (p < 0.05 significance).
   - __Key Metrics:__ Revenue per category, conversion rates, profit margins.
## Data Used
📊 __Dataset:__ [Retail Sales Data](https://github.com/ackben0226/Price-Strategy-Using-A-B-Testing/blob/main/Retail%20Sales%20Data.csv) (1,000+ transactions)

📌 __Features:__
   - Product Category (Electronics, Beauty, Home, etc.)
   - Price per Unit
   - Quantity Sold
   - Total Amount
   - new features: __Adjusted Revenue:__ = ___Quantity Sold___ __*__ ___Adjusted Price___.
   
## 2. Key Results & Visualization
### A) Discount Strategy (10% Off)
__Impact on Revenue by Category:__
|Category|	Avg Revenue (No Discount)|	Avg Revenue (10% Off)|	P-value|
|---------|--------|--------|-------|
|Electronics|	$2,955.06|	$2,576.84|	0.050|
|Clothing|	$1597.40|	$1396.55|	0.0079|
|Beauty	|$148.21|$145.52	|0.3506|

📌 **Insight:** Sports, Clothing, Electronics, and Home all show significant drops in average revenue per order, except Home that maintains total revenue—others lose both per-unit and total income. Beauty and Books show no significant impact—volume helped raise total revenue, but the profit per sale declined slightly.

![image](https://github.com/user-attachments/assets/55047acf-1338-43e2-ba5d-8308fba7f933)

### B) Price Increase (10% Higher)
__Impact on Revenue__
|Group	|Avg Revenue	|Sales Volume Change|
|-----|----|-----|
|Original Price	|$1,106.59	|Baseline|
|10% Increase	|$1,165.05|	↓0.98%|

📌 __Insight:__ Price hikes boost revenue but demand drops slightly.

### C) Bundling Strategy
__Revenue Comparison:__
|Group	|Avg Order Value|
|-----|----|
|Individual	|$11,395.22|
|Bundle Offer	|$12,996.22 (+14%)|

📌 __Insight:__ Bundling increases order value significantly.

![image](https://github.com/user-attachments/assets/589a40a8-95de-46b4-a879-ff8d86322dd2)

## 3. Recommendations
### 1. Optimize Discounts
- Discount may hurt Electronics, Clothing & Sports
- Limited discount in: Books, Beauty & Home (maintained revenue despite margin pressure).
  
### 2. Test Price Increases Carefully
- Start with premium categories (Electronics, Home).
- Monitor churn risk in price-sensitive segments.

### 3. Expand Bundling Strategies
- Pair frequently bought together (e.g., Sports + Home).
- Run promotions ("Buy X, Get Y at 10% off").

###  4. Future Enhancements
🔹 **Machine Learning:** Predict optimal pricing per customer segment.
🔹 **Dynamic Pricing:** Adjust in real-time based on demand.
🔹 **Seasonal Testing:** Compare holiday vs. regular pricing.

## __Results Summary__
- __Discount Strategy:__ The 10% discount increases revenue in some categories and vice-versa in other categories. It also leads to a decline in profit across all categories
- __Price Increase Sensitivity:__ The 10% price increase resulted in a ~1% (0.98%) drop in sales volume but a 5.28% increase in revenue, indicating that customers were relatively price-insensitive.
- __Bundling Strategy:__ Bundled products saw a 14% higher average order value compared to individual item sales, demonstrating the effectiveness of bundling in driving higher revenue.

## 4. Conclusion
**Pricing strategy significantly impacts profitability.**
- Discounts drive volume but may hurt margins.
- Price increases can boost revenue if applied strategically.
- Bundling enhances average order value.

**Next Steps:**
- __Deploy real-time A/B tests__ in production.
- __Refine segmentation__ (e.g., loyalty vs. new customers).

📂 __GitHub Code:__ [Price Strategy Using A/B Testing](https://github.com/ackben0226/Price-Strategy-Using-A-B-Testing/blob/main/Price_Strategy_Using_A_B_Testing.ipynb)

**🚀By leveraging data-driven pricing, businesses can maximize revenue while maintaining customer satisfaction.** 



