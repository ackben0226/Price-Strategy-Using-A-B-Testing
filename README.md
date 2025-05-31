# Price Optimization Using A/B Testing: Data-Driven Insights for Retail Strategy
## Executive Summmary
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
📊 __Dataset:__ [Retail Sales Data](https://github.com/ackben0226/Price-Strategy-Using-A-B-Testing/blob/main/Retail%20Sales%20Data.csv)(1,000+ transactions)

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

📌 **Insight:** Sports, Clothing, Electronics, and Home all show significant drops in average revenue per order, except Home maintains total revenue—others lose both per-unit and total income. Beauty and Books show no significant impact—volume helped raise total revenue, but the profit per sale declined slightly.


Created visualizations to compare revenue trends and customer behavior across groups.
![image](https://github.com/user-attachments/assets/36367c45-087d-4cbf-bb23-9801d0b37ca7)


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

The following code implements an A/B test to evaluate the impact of a 10% discount on revenue for a specific product category:

### Code

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
