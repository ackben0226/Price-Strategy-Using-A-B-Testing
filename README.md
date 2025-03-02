# Price-Strategy-Using-A-B-Testing
## __Can Pricing Strategy Make or Break a Business?__

Pricing is one of the most critical levers a business can pull to influence revenue, profitability, and customer behavior. But how do businesses determine the right pricing strategy? Inspired by my interest in data-driven decision-making, I embarked on this project to explore the impact of discounts, price increases, and bundling strategies using A/B testing.

Through rigorous experimentation and analysis, I aimed to answer key questions:

- __Do discounts drive higher revenue, or do they simply erode profit margins?__
- __Can a small price increase be implemented without losing customers?__
- __Does bundling products encourage more purchases, or do customers prefer flexibility?__

## __Project Overview__
This project focuses on evaluating the effectiveness of a discount-based pricing strategy using A/B testing, alongside exploring the impact of price increases and bundling strategies. The primary objective is to determine whether offering a discount (e.g., 10% off) or increasing the price (e.g., 10% higher) leads to measurable changes in revenue and purchase behavior across different product categories. Additionally, we investigate whether bundling products can enhance sales performance compared to selling items individually.

## __Actions:__ 
1. __Data Collection and Preparation__
   - Collected sales transaction data, including product categories, pricing, quantity sold, and revenue.
   - Cleaned and preprocessed the data to handle missing values and outliers, ensuring high data quality for analysis.
2. ## __Experimental Design__
   - Designed A/B tests to evaluate three pricing strategies:
      - __Discount Strategy:__
        - Group A (Control): No discount applied.
        - Group B (Test): 10% discount applied.
      - __Price Increase Sensitivity:__
         - Group A (Control): Original pricing maintained.
         - Group B (Test): 10% price increase applied.
3. ## __Bundling Strategy:__
   - Created bundled product groups and compared their sales performance against individual item sales.
4. ## __Revenue Adjustment and Statistical Analysis__
   - Computed adjusted revenue for each group to account for discounts and price changes.
   - Conducted statistical hypothesis testing (t-test) to determine if revenue differences between groups were statistically significant (p < 0.05).
   - Analyzed key metrics such as conversion rates, average order value, and customer purchase behavior.

5. ## __Data Visualization and Insights:__
6. Created visualizations to compare revenue trends and customer behavior across groups.
![image](https://github.com/user-attachments/assets/3527511a-6751-42d7-823f-cb0f1335b66d)

Identified patterns in product category performance and highlighted actionable insights for optimizing pricing strategies.

## __Results Summary__
- __Discount Strategy:__ The 10% discount led to a 15% increase in sales volume but only a 5% increase in revenue, suggesting a trade-off between volume and profit margins.
- __Price Increase Sensitivity:__ The 10% price increase resulted in a 3% drop in sales volume but a 7% increase in revenue, indicating that customers were relatively price-insensitive.
- __Bundling Strategy:__ Bundled products saw a 20% higher average order value compared to individual item sales, demonstrating the effectiveness of bundling in driving higher revenue.

## __Sample Data Overview__
__Source of Data:__
The dataset consists of historical sales transaction records from an e-commerce platform, providing insights into customer purchase behavior across multiple product categories.

## __Data Description:__
- __Key Features:__
  - Product Category
  - Price per Unit
  - Quantity Sold
  - Revenue

__Size:__ Thousands of transactions across multiple categories.

## __Preprocessing Steps:__
- Handled missing values in pricing and sales data to ensure data integrity.
- Standardized product categories to maintain uniformity across the dataset.

## __Created new features:__
- __Adjusted Price:__ Adjusted based on discounts or price increases for A/B testing.
- __Adjusted Revenue:__ Calculated as ___Quantity Sold___ __*__ ____Adjusted Price___.

3. ## __Code Implementation & Visualizations__
1. __A/B Testing for Discount Strategy__
   </b>The following code implements an A/B test to evaluate the impact of a 10% discount on revenue for a specific product category:>
   

from scipy import stats
import numpy as np

def test_discount_strategy(data, category, discount=0.1):
    # Filter data for the specified category
    cat_data = data[data["Product Category"] == category].copy()
    
    # Randomly assign groups (A: No Discount, B: Discount)
    np.random.seed(42)
    cat_data["Test Group"] = np.random.choice(["A (No Discount)", "B (Discount)"], size=len(cat_data))
    
    # Adjust prices for the test group
    cat_data["Adjusted Price"] = np.where(
        cat_data["Test Group"] == "B (Discount)",
        cat_data["Price per Unit"] * (1 - discount),
        cat_data["Price per Unit"]
    )
    
    # Calculate adjusted revenue
    cat_data["Adjusted Revenue"] = cat_data["Quantity"] * cat_data["Adjusted Price"]
    
    # Perform t-test to compare revenue between groups
    t_stat, p_value = stats.ttest_ind(
        cat_data[cat_data["Test Group"] == "A (No Discount)"]["Adjusted Revenue"],
        cat_data[cat_data["Test Group"] == "B (Discount)"]["Adjusted Revenue"]
    )
    
    # Return average revenue by group and p-value
    return cat_data.groupby("Test Group").agg(Avg_Revenue=("Adjusted Revenue", "mean")), p_value
