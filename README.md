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
**Objective**
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

Created visualizations to compare revenue trends and customer behavior across groups.
![image](https://github.com/user-attachments/assets/36367c45-087d-4cbf-bb23-9801d0b37ca7)

## 3. Recommendations
### 1. Optimize Discounts
- Discount may hurt Electronics, Clothing & Sports
- Limited discount in: Books, Beauty & Home (maintained revenue despite margin pressure).
  
### 2. Test Price Increases Carefully
- Start with premium categories (Electronics, Home).
- Monitor churn risk in price-sensitive segments.

### 3. Expand Bundling Strategies
- Pair frequently bought together (e.g., Sports + Home).
_ Run promotions ("Buy X, Get Y at 10% off").

###  4. Future Enhancements
🔹 **Machine Learning:** Predict optimal pricing per customer segment.
🔹 **Dynamic Pricing:** Adjust in real-time based on demand.
🔹 **Seasonal Testing:** Compare holiday vs. regular pricing.

## 4. Conclusion
**Pricing strategy significantly impacts profitability.**
- Discounts drive volume but may hurt margins.
- Price increases can boost revenue if applied strategically.
- Bundling enhances average order value.

**Next Steps:**
- __Deploy real-time A/B tests__ in production.
- __Refine segmentation__ (e.g., loyalty vs. new customers).

📂 GitHub Code: Price Strategy Using A/B Testing

**By leveraging data-driven pricing, businesses can maximize revenue while maintaining customer satisfaction.** 🚀


Identified patterns in product category performance and highlighted actionable insights for optimizing pricing strategies.

## __Results Summary__
- __Discount Strategy:__ The 10% discount led to a 15% increase in sales volume but only a 5% increase in revenue, suggesting a trade-off between volume and profit margins.
- __Price Increase Sensitivity:__ The 10% price increase resulted in a 3% drop in sales volume but a 7% increase in revenue, indicating that customers were relatively price-insensitive.
- __Bundling Strategy:__ Bundled products saw a 20% higher average order value compared to individual item sales, demonstrating the effectiveness of bundling in driving higher revenue.

## 2. __Data Overview__

__Data:__ [Retail Sales Data](https://github.com/ackben0226/Price-Strategy-Using-A-B-Testing/blob/main/Retail%20Sales%20Data.csv)(1,000+ transactions)


## __Preprocessing Steps:__
- Handled missing values in pricing and sales data to ensure data integrity.
- Standardized product categories to maintain uniformity across the dataset.

## __Created new features:__
- __Adjusted Price:__ Adjusted based on discounts or price increases for A/B testing.
- __Adjusted Revenue:__ Calculated as ___Quantity Sold___ __*__ ___Adjusted Price___.

## 3. __Code Implementation & Visualizations__ 
The Python code used to conduct the A/B tests, perform data analysis, and generate results is hosted on GitHub. You can explore the code and run the analysis yourself:

__GitHub Repository:__  [Price Strategy Using A/B Testing](https://github.com/ackben0226/Price-Strategy-Using-A-B-Testing/blob/main/Price_Strategy_Using_A_B_Testing.ipynb) - GitHub
### A/B Testing for Discount Strategy



## 4. __Executive Summary__
- __Objective Recap:__
This project evaluates the impact of discount-based pricing strategies, price increases, and bundling on revenue using A/B testing, helping businesses make data-driven decisions.
- __Key Findings:__
  - Discounts positively impacted revenue for certain product categories.
  - Price increases showed mixed effects, with some categories retaining customers while others experienced a drop in revenue.
  - Bundling strategies improved sales performance for complementary products.
- __Challenges & Learnings:__
  - Some categories did not exhibit revenue improvement, suggesting price elasticity varies across products.
  - A more refined segmentation approach could improve pricing strategies.

## 5. __Discussion, Growth, and Next Steps__
- __Future Enhancements:__
  - Introduce machine learning models to predict optimal discount and pricing strategies.
  - Perform multivariate testing to analyze the impact of other factors (e.g., seasonal trends, customer segmentation).
- __Deployment Considerations:__
  - Implement real-time A/B testing in production environments.
  - Automate data collection and analysis pipelines for continuous monitoring.
- __Expanding the Scope:__
  - Test pricing strategies across different customer demographics.
  - Explore additional promotional strategies (e.g., bundling, limited-time offers) to increase sales.
    
This project lays the foundation for data-driven pricing optimization, equipping businesses with actionable insights to maximize revenue while maintaining customer satisfaction.
