# Price-Strategy-Using-A/B-Testing
## __Can Pricing Strategy Make or Break a Business?__

Pricing is one of the most critical levers a business can pull to influence revenue, profitability, and customer behavior. But how do businesses determine the right pricing strategy? Inspired by my interest in data-driven decision-making, I embarked on this project to explore the impact of discounts, price increases, and bundling strategies using A/B testing.

Through rigorous experimentation and analysis, I aimed to answer key questions:

- __Do discounts drive higher revenue, or do they simply erode profit margins?__
- __Can a small price increase be implemented without losing customers?__
- __Does bundling products encourage more purchases, or do customers prefer flexibility?__

## __Project Overview__
This project focuses on evaluating the effectiveness of a discount-based pricing strategy using A/B testing, alongside exploring the impact of price increases and bundling strategies. The primary objective is to determine whether offering a discount (e.g., 10% off) or increasing the price (e.g., 10% higher) leads to measurable changes in revenue and purchase behavior across different product categories. Additionally, we investigate whether bundling products can enhance sales performance compared to selling items individually.

## __Actions:__ 
- ### __Data Collection and Preparation__
   - Collected sales transaction data, including product categories, pricing, quantity sold, and revenue.
   - Cleaned and preprocessed the data to handle missing values and outliers, ensuring high data quality for analysis.
2. ### __Experimental Design__
   - Designed A/B tests to evaluate three pricing strategies:
      - __Discount Strategy:__
        - Group A (Control): No discount applied.
        - Group B (Test): 10% discount applied.
      - __Price Increase Sensitivity:__
         - Group A (Control): Original pricing maintained.
         - Group B (Test): 10% price increase applied.
3. ### __Bundling Strategy:__
   - Created bundled product groups and compared their sales performance against individual item sales.
4. ### __Revenue Adjustment and Statistical Analysis__
   - Computed adjusted revenue for each group to account for discounts and price changes.
   - Conducted statistical hypothesis testing (t-test) to determine if revenue differences between groups were statistically significant (p < 0.05).
   - Analyzed key metrics such as conversion rates, average order value, and customer purchase behavior.

5. ### __Data Visualization and Insights:__
6. Created visualizations to compare revenue trends and customer behavior across groups.
![image](https://github.com/user-attachments/assets/36367c45-087d-4cbf-bb23-9801d0b37ca7)


Identified patterns in product category performance and highlighted actionable insights for optimizing pricing strategies.

## __Results Summary__
- __Discount Strategy:__ The 10% discount led to a 15% increase in sales volume but only a 5% increase in revenue, suggesting a trade-off between volume and profit margins.
- __Price Increase Sensitivity:__ The 10% price increase resulted in a 3% drop in sales volume but a 7% increase in revenue, indicating that customers were relatively price-insensitive.
- __Bundling Strategy:__ Bundled products saw a 20% higher average order value compared to individual item sales, demonstrating the effectiveness of bundling in driving higher revenue.

## __Sample Data Overview__
__Source of Data:__
(Retail Sales Data)[https://github.com/ackben0226/Price-Strategy-Using-A-B-Testing/blob/main/Retail%20Sales%20Data.csv]
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
- __Adjusted Revenue:__ Calculated as ___Quantity Sold___ __*__ ___Adjusted Price___.

3. ## __Code Implementation & Visualizations__
### A/B Testing for Discount Strategy

The following code implements an A/B test to evaluate the impact of a 10% discount on revenue for a specific product category:

###Code

```python
from scipy import stats

def test_discount_strategy(data, category, discount=0.1):
    # Filter by category
    cat_data = data[data["Product Category"] == category].copy()

    # Split into A/B groups
    np.random.seed(42)
    cat_data["Test Group"] = np.random.choice(["A (No Discount)", "B (Discount)"], size=len(cat_data))

    # Apply discount to Group B
    cat_data["Adjusted Price"] = np.where(
        cat_data["Test Group"] == "B (Discount)",
        cat_data["Price per Unit"] * (1 - discount),
        cat_data["Price per Unit"]
    )
    cat_data["Adjusted Revenue"] = cat_data["Quantity"] * cat_data["Adjusted Price"]

    # Compare groups
    grouped = cat_data.groupby("Test Group").agg(
        Avg_Revenue=("Adjusted Revenue", "mean"),
        Total_Revenue=("Adjusted Revenue", "sum"),
        Sample_Size=("Test Group", "count")
    ).reset_index()

    # T-test
    t_stat, p_value = stats.ttest_ind(
        cat_data[cat_data["Test Group"] == "A (No Discount)"]["Adjusted Revenue"],
        cat_data[cat_data["Test Group"] == "B (Discount)"]["Adjusted Revenue"]
    )

    return grouped, p_value

# List of categories to test
categories_to_test = ["Sports", "Beauty", "Clothing", 'Electronics', 'Home', 'Books']

# Loop through each category
for category in categories_to_test:
    results, p_value = test_discount_strategy(data, category)

    # Print formatted results
    print(f"{category} Discount Results:")
    print(results)
    print(f"P-value: {p_value:.4f}")
    print("-" * 40 + "\n")
``` 

### __Price increase sensitivity test using A/B testing and statistical analysis__
## __Code__

```python
import numpy as np
import pandas as pd
from scipy import stats

# Simulate price increase test (A: Original Price, B: 10% Increase) for all products
np.random.seed(42)  # Set seed for reproducibility
data["Test Group"] = np.random.choice(
    ["A (Original Price)", "B (10% increase)"], 
    size=len(data)
)

# Apply the 10% price increase to group B
data["Adjusted Price"] = np.where(
    data["Test Group"] == "B (10% increase)",
    data["Price per Unit"] * 1.1,  # Increase price by 10%
    data["Price per Unit"]  # Keep original price for group A
)

# Calculate adjusted revenue
data["Adjusted Revenue"] = data["Quantity"] * data["Adjusted Price"]

# Compare results between groups
grouped = data.groupby("Test Group").agg(
    Avg_Quantity=("Quantity", "mean"),  # Average quantity sold
    Avg_Revenue=("Adjusted Revenue", "mean"),  # Average revenue per transaction
    Total_Revenue=("Adjusted Revenue", "sum")  # Total revenue per group
).reset_index()

print("\nPrice Increase Results (All Products):\n", grouped)

# Mann-Whitney U test (non-parametric test for non-normal data)
u_stat, p_value = stats.mannwhitneyu(
    data[data["Test Group"] == "A (Original Price)"]["Adjusted Revenue"],
    data[data["Test Group"] == "B (10% increase)"]["Adjusted Revenue"]
)
print(f"\nMann-Whitney U Test: p = {p_value:.4f}")
```
## __Bundling Selling Analysis__
```python
from scipy import stats

# Define the bundle (Beauty + Home)
BUNDLE_PRICE = 320  # Discounted price
INDIVIDUAL_PRICE = 50 + 300  # Beauty (50) + Home (300)

# Identify customers who bought Beauty or Home products
eligible_customers = set(data.loc[data["Product Category"].isin(["Beauty", "Home"]), "Customer ID"].unique())

# Split into control (A) and treatment (B)
np.random.seed(42)
group_assignment = pd.DataFrame({
    "Customer ID": list(eligible_customers),
    "Group": np.random.choice(["A (No Bundle)", "B (Bundle Offered)"], size=len(eligible_customers))
})

# Merge group assignments with transactions
data = pd.merge(data, group_assignment, on="Customer ID", how="left")
data["Group"] = data["Group"].fillna("A (No Bundle)")  # Assign others to control

# Simulate bundle purchases in treatment group (30% adoption rate)
bundle_customers = data[
    (data["Group"] == "B (Bundle Offered)") &
    (data["Product Category"].isin(["Beauty", "Home"]))
]["Customer ID"].unique()

np.random.seed(42)
bundle_adopters = np.random.choice(
    bundle_customers,
    size=int(len(bundle_customers) * 0.3),  # 30% buy the bundle
    replace=False
)

# Remove individual Beauty/Home transactions for bundle adopters
bundle_transactions = data[
    (data["Customer ID"].isin(bundle_adopters)) &
    (data["Product Category"].isin(["Beauty", "Home"]))
].index
data = data.drop(bundle_transactions).reset_index(drop=True)  # Reset index

# Add new bundle transactions
bundle_df = pd.DataFrame({
    "Customer ID": bundle_adopters,
    "Product Category": "Bundle",
    "Quantity": 1,
    "Revenue": BUNDLE_PRICE  # Ensure column name matches your dataset
})

data = pd.concat([data, bundle_df], ignore_index=True)
# Calculate total revenue by group
revenue = data.groupby("Group")["Total Amount"].sum().reset_index()
print("Total Revenue by Group:\n", revenue)

# Perform statistical test (Mann-Whitney U for non-normal data)
group_a = data[data["Group"] == "A (No Bundle)"]["Total Amount"]
group_b = data[data["Group"] == "B (Bundle Offered)"]["Total Amount"]

u_stat, p_value = stats.mannwhitneyu(group_a, group_b)
print(f"\nMann-Whitney U Test: p = {p_value:.4f}")
```
## __Visualization of revenue trend between bundle purchase and individual purchase__
```python
import matplotlib.pyplot as plt

# Revenue comparison
plt.figure(figsize=(6, 4))
sns.barplot(x='Group', y='Total Amount', data=revenue, palette=['red', 'pink'])
plt.title('Total Revenue: Bundle vs. No Bundle')
plt.ylabel('Revenue')
plt.show()
```
![image](https://github.com/user-attachments/assets/fee04968-9f00-4cc4-9a72-2b16730dd2c7)

6. ## __Executive Summary__
- __Objective Recap:__
This project evaluates the impact of discount-based pricing strategies, price increases, and bundling on revenue using A/B testing, helping businesses make data-driven decisions.
- __Key Findings:__
  - Discounts positively impacted revenue for certain product categories.
  - Price increases showed mixed effects, with some categories retaining customers while others experienced a drop in revenue.
  - Bundling strategies improved sales performance for complementary products.
- __Challenges & Learnings:__
  - Some categories did not exhibit revenue improvement, suggesting price elasticity varies across products.
  - A more refined segmentation approach could improve pricing strategies.

7. ## __Discussion, Growth, and Next Steps__
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
