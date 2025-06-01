import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

# Load dataset
url = 'https://raw.githubusercontent.com/ackben0226/Price-Strategy-Using-A-B-Testing/main/Retail%20Sales%20Data.csv'
data = pd.read_csv(url)

# Product categories
categories = data["Product Category"].unique()

def assign_realistic_costs(data, cost_margin=0.3):
    data = data.copy()
    data["Cost per Unit"] = data["Price per Unit"] * cost_margin
    return data

# A/B testing simulation
def test_discount_strategy(data, category, discount=0.1):
    # Ensure costs are calculated
    data = assign_realistic_costs(data)

    # Filter customers who bought the selected category
    customers = data[data["Product Category"] == category]["Customer ID"].unique()

    # Randomly assign test groups
    np.random.seed(42)
    customer_groups = pd.DataFrame({
        "Customer ID": customers,
        "Test Group": np.random.choice(["A (No Discount)", "B (Discount)"], size=len(customers))
    })

    # Merge group assignment with main data
    cat_data = pd.merge(data, customer_groups, on="Customer ID", how="inner")

    # Apply discount logic
    cat_data["Adjusted Unit Price"] = cat_data.apply(
        lambda row: row["Price per Unit"] * (1 - discount) if row["Test Group"] == "B (Discount)" else row["Price per Unit"],
        axis=1
    )

    # Calculate adjusted revenue and profit
    cat_data["Adjusted Revenue"] = cat_data["Adjusted Unit Price"] * cat_data["Quantity"]
    cat_data["Adjusted Profit"] = (cat_data["Adjusted Unit Price"] - cat_data["Cost per Unit"]) * cat_data["Quantity"]

    # Aggregate results
    grouped = cat_data.groupby("Test Group").agg(
        Total_Profit=("Adjusted Profit", "sum"),
        Total_Revenue=("Adjusted Revenue", "sum"),
        Average_Profit_Per_Customer=("Adjusted Profit", "mean"),
        Average_Sales=("Adjusted Revenue", "mean")  # Added for consistency with original results_df
    ).reset_index()

    # T-test for profit difference
    profit_A = cat_data[cat_data["Test Group"] == "A (No Discount)"]["Adjusted Profit"]
    profit_B = cat_data[cat_data["Test Group"] == "B (Discount)"]["Adjusted Profit"]
    t_stat, p_value = stats.ttest_ind(profit_A, profit_B, alternative='two-sided')

    return grouped, p_value

# Precompute test results
categories_to_test = ["Sports", "Beauty", "Clothing", 'Electronics', 'Home', 'Books']
results = []
all_results = []

for category in categories_to_test:
    grouped_results, p_value = test_discount_strategy(data, category=category, discount=0.1)
    grouped_results["Category"] = category
    grouped_results["p_value"] = p_value
    all_results.append(grouped_results)

    # For results_df (used in visualizations)
    for _, row in grouped_results.iterrows():
        results.append({
            "Category": category,
            "Test Group": row["Test Group"],
            "Average Sales": row["Average_Sales"]
        })

results_df = pd.DataFrame(results)
final_results_df = pd.concat(all_results, ignore_index=True)

# Melt the dataframe for plotting
plot_data = final_results_df.melt(
    id_vars=["Category", "Test Group", "p_value"],
    value_vars=["Total_Profit", "Total_Revenue"],
    var_name="Metric",
    value_name="Amount"
)

from itertools import combinations

# Function to determine bundle prices
def determine_bundle_prices(discount=0.15):
    # Average prices per category
    avg_unit_price = data.groupby('Product Category')['Price per Unit'].mean()
    prices = avg_unit_price[avg_unit_price.index != 'Bundle']

    # Generate all unique category pairs
    category_pairs = list(combinations(prices.index, 2))

    # Calculate bundle prices
    bundle_prices = {}
    for cat1, cat2 in category_pairs:
        individual_total = prices[cat1] + prices[cat2]
        bundle_price = round(individual_total * (1 - discount), 2)
        bundle_prices[(cat1, cat2)] = (round(individual_total, 2), bundle_price)

    return bundle_prices

# Compute bundle prices
bundle_prices = determine_bundle_prices()

# Generate category pairs for dropdown
category_pairs = list(combinations(categories_to_test, 2))
pair_options = [{"label": f"{pair[0]}+{pair[1]}", "value": f"{pair[0]}+{pair[1]}"}
                for pair in category_pairs]

# Identify eligible customers (last 30 days from latest date)
data["Date"] = pd.to_datetime(data["Date"])
latest_date = data["Date"].max()
eligible_customers = data[
    data["Date"] >= (latest_date - pd.Timedelta(days=30))
]["Customer ID"].unique()

# Split into control and treatment
np.random.seed(42)
group_assignment = pd.DataFrame({
    "Customer ID": list(eligible_customers),
    "Group": np.random.choice(["A (No Bundle)", "B (Bundle Offered)"], size=len(eligible_customers))
})

# Merge assignments into main data
data = pd.merge(data, group_assignment, on="Customer ID", how="left", suffixes=('', '_assigned'))

# Compute price sensitivity
data["Price Sensitivity"] = np.clip(
    (data["Price per Unit"] - data["Price per Unit"].mean()) / data["Price per Unit"].std(),
    -2, 2
)

# Simulate bundle pricing
bundle_prices = determine_bundle_prices()
BUNDLE_PRICE = np.mean([price[1] for price in bundle_prices.values()])  # average discounted bundle price

# Estimate adoption in treatment group
treatment_mask = data["Group"] == "B (Bundle Offered)"
base_conversion_rate = 0.10  # average bundle adoption

customer_conversion_rates = data.loc[
    treatment_mask,
    ["Customer ID", "Price Sensitivity"]
].groupby("Customer ID").mean().reset_index()

customer_conversion_rates["Conversion Rate"] = base_conversion_rate * (1 - 0.1 * customer_conversion_rates["Price Sensitivity"])

# Defensive check for non-empty
if not customer_conversion_rates.empty:
    adoption_prob = customer_conversion_rates["Conversion Rate"].values

    np.random.seed(42)
    bundle_adopters = np.random.choice(
        customer_conversion_rates["Customer ID"].values,
        size=int(len(customer_conversion_rates) * adoption_prob.mean()),
        replace=False,
        p=adoption_prob / adoption_prob.sum()
    )
else:
    print("⚠️ No eligible customers for bundle adoption in Treatment group.")
    bundle_adopters = []

# Remove individual purchases from adopters
individual_purchases_mask = data["Customer ID"].isin(bundle_adopters)
data = data[~individual_purchases_mask]

# Add synthetic bundle transactions
bundle_transactions = pd.DataFrame({
    "Transaction ID": [f"BUNDLE_{x}" for x in range(len(bundle_adopters))],
    "Date": latest_date + pd.Timedelta(days=1),
    "Customer ID": bundle_adopters,
    "Product Category": "Bundle",
    "Quantity": 1,
    "Price per Unit": BUNDLE_PRICE,
    "Total Amount": BUNDLE_PRICE,
    "Group": "B (Bundle Offered)"
})

# Add missing columns
for col in data.columns:
    if col not in bundle_transactions.columns:
        bundle_transactions[col] = np.nan

# Align column order
bundle_transactions = bundle_transactions[data.columns]
data = pd.concat([data, bundle_transactions], ignore_index=True)

# Sort, tag, and analyze
data = data.sort_values("Date")
data["Bundled"] = np.where(data["Product Category"] == "Bundle", "Yes", "No")

# Revenue per customer
revenue_per_customer = data.groupby(["Customer ID", "Group"])["Total Amount"].sum().reset_index()

# Revenue totals
a_rev = revenue_per_customer[revenue_per_customer["Group"] == "A (No Bundle)"]["Total Amount"].sum()
b_rev = revenue_per_customer[revenue_per_customer["Group"] == "B (Bundle Offered)"]["Total Amount"].sum()

# Revenue lift
lift = (b_rev - a_rev) / a_rev * 100

# Statistical significance
t_stat, p_value = stats.ttest_ind(
    revenue_per_customer[revenue_per_customer["Group"] == "A (No Bundle)"]["Total Amount"],
    revenue_per_customer[revenue_per_customer["Group"] == "B (Bundle Offered)"]["Total Amount"],
    equal_var=False
)

def analyze_price_impact(data, price_increase=0.10, random_seed=42):
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # 1. Experimental Setup
    data["Test Group"] = np.random.choice(
        ["A (Original Price)", "B (10% Price Increase)"], 
        size=len(data),
        p=[0.5, 0.5]  # Equal split for A/B test
    )

    # 2. Apply Price Adjustment
    data["Adjusted Price"] = np.where(
        data["Test Group"] == "B (10% Price Increase)",
        data["Price per Unit"] * (1 + price_increase),
        data["Price per Unit"]
    )

    # 3. Calculate Metrics
    data["Adjusted Revenue"] = data["Quantity"] * data["Adjusted Price"]

    # 4. Aggregate Results
    grouped = data.groupby("Test Group").agg(
        Avg_Quantity=("Quantity", "mean"),
        Avg_Revenue=("Adjusted Revenue", "mean"),
        Revenue_Lift=("Adjusted Revenue", lambda x: x.sum()/data[data["Test Group"]=="A (Original Price)"]["Adjusted Revenue"].sum() - 1)
    ).reset_index()

    # 5. Statistical Testing
    # Mann-Whitney for revenue comparison (non-parametric)
    u_stat, p_value = stats.mannwhitneyu(
        data[data["Test Group"] == "A (Original Price)"]["Adjusted Revenue"],
        data[data["Test Group"] == "B (10% Price Increase)"]["Adjusted Revenue"],
        alternative='two-sided'
    )

    return grouped, p_value

grouped, p_value = analyze_price_impact(data)

for _, row in grouped.iterrows():
    print(f"\n--- {row['Test Group']} ---")
    print(f"Average Quantity: {row['Avg_Quantity']:.2f}")
    print(f"Average Revenue: £{row['Avg_Revenue']:.2f}")
    print(f"Revenue Lift: {row['Revenue_Lift']*100:.2f}%")

# Optionally print the statistical test result
print(f"\nMann-Whitney U Test p-value: {p_value:.4f}")

# App initialization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server
app.title = "A/B Price Strategy Dashboard"

#def summary_card(title1, value):
    #return dbc.Card([
        #dbc.CardBody([
            #html.H4(title, className="card-title"),
            #html.P(value, className="card-text")
        #])
    #], className="mb-3")

def summary_card(title1, value1, title2=None, value2=None, title3=None, value3=None, 
                 title4=None, value4=None, title5=None, value5=None):
    # Right content column
    body = [
        html.H5(title1, className="card-title"),
        html.P(value1, className="card-text fw-bold")
    ]

    if title2 and value2:
        body += [
            html.Hr(style={"borderTop": "1px solid #ccc", "margin": "10px 0"}),
            html.H5(title2, className="card-title"),
            html.P(value2, className="card-text fw-bold")
        ]
    if title3 and value3:
        body += [
            html.Hr(style={"borderTop": "1px solid #ccc", "margin": "10px 0"}),
            html.H5(title3, className="card-title"),
            html.P(value3, className="card-text fw-bold")
        ]
    if title4 and value4:
        body += [
            html.Hr(style={"borderTop": "1px solid #ccc", "margin": "10px 0"}),
            html.H5(title4, className="card-title"),
            html.P(value4, className="card-text fw-bold")
        ]
    if title5 and value5:
        body += [
            html.Hr(style={"borderTop": "1px solid #ccc", "margin": "10px 0"}),
            html.H5(title5, className="card-title"),
            html.P(value5, className="card-text fw-bold")
        ]

    return dbc.Card([
        dbc.CardBody(body)
    ], className="mb-3")

# App layout
app.layout = dbc.Container([
    html.H2("A/B Testing Price Strategy Dashboard", className="text-center my-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Product Category:"),
            dcc.Dropdown(
                id="category-dropdown-1",
                options=[{"label": "All", "value": "All"}] + [{"label": cat, "value": cat} for cat in categories],
                value="All"
            ) 
        ]),
    ]),
    html.Hr(),
    dcc.Tabs(id="tabs", value="tab_summary", children=[
        dcc.Tab(label="Summary", value="tab_summary",
               style={'backgroundColor': '#ffa500', 'color': 'white', 'fontWeight': 'bold', 'padding': '10px',
                     'borderRadius': '5px', 'text-align' : 'center'},
               selected_style={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold',
                               'padding': '10px'}),
        dcc.Tab(label="Data Table", value="tab_table",
               style={'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'}),
        dcc.Tab(label="A/B Revenue Comparison", value="tab_ab_revenue",
                style={'backgroundColor': '#ffd700', 'color': 'white', 'fontWeight': 'bold'}),
        dcc.Tab(label="Price Optimization", value="tab_price_optimization",
                style={'backgroundColor': '#ffc0cb', 'color': 'white', 'fontWeight': 'bold'}),
        dcc.Tab(label="Customer Segmentation", value="tab_segmentation",
                style={'backgroundColor': '#32cd32', 'color': 'white', 'fontWeight': 'bold'}),
        dcc.Tab(label="A/B Visualizations", value="tab_visualize",
               style={'backgroundColor': '#ff0000', 'color': 'white', 'fontWeight': 'bold'})
    ]),
    html.Div(id="tab-content"),
    html.Div([
        html.Label('Select Bundle Pair:'),
        dcc.Dropdown(
            id="category-dropdown-2",
            options=[{"label": "All", "value": "All"}] + pair_options,
            value="All")
    ], id="dropdown2-container", style={"display": "none"}),
    html.Div(id="fig5-container")  # Container for fig5 content
])
# Callbacks
@app.callback(
    Output("tab-content", "children"),
    Output("dropdown2-container", "style"),
    Input("tabs", "value"),
    Input("category-dropdown-1", "value")    
)
def render_content(tab, selected_category_1):
    # Show/hide the second dropdown based on the selected tab
    dropdown2_style = {"display": "block"} if tab == "tab_visualize" else {"display": "none"}

    selected_category = selected_category_1
    filtered_data = data if selected_category == "All" else data[data["Product Category"] == selected_category]

    if filtered_data.empty:
        return html.Div("No data available for this category."), dropdown2_style

    if tab == 'tab_summary':
        total = filtered_data['Total Amount'].sum()
        avg = filtered_data['Total Amount'].mean()
        std = filtered_data['Total Amount'].std()

        stat_info = None
        if selected_category != "All":
            try:
                ab_results = final_results_df[
                    (final_results_df["Category"] == selected_category)
                    & (final_results_df["Test Group"] == "B (Discount)")
                ]
                stat_p = ab_results["p_value"].values[0]
                significance = "Significant" if stat_p < 0.05 else "Not Significant"
                stat_info = summary_card("Statistical Significance (p-value)", f"{stat_p:.4f} → {significance}")
            except IndexError:
                stat_info = summary_card("Statistical Significance", "Not available")

        return dbc.Row([
            dbc.Col(html.H5(summary_card("Total Revenue", f"${total:,.2f}")
                           ), width=4),
            dbc.Col(html.H5(summary_card("Average Revenue per Order", f"${avg:,.2f}")
                   ), width=4),
            dbc.Col(html.H5(summary_card("Standard Deviation", f"${std:,.2f}")
                           ), width=4),
            dbc.Col(html.H5(
                summary_card(
                    "No Bundle Revenue", f"${a_rev:,.0f}", "Bundle Offered Revenue", f"${b_rev:,.0f}", 
                             "Revenue Lift", f"+{lift:,.1f}%", "Revenue Impact")),width=4
                ),
            dbc.Col(html.H5(
                summary_card(
                    "Test Group", f"{row['Test Group']}",
                    "Average Quantity", f"{row['Avg_Quantity']:.2f}",
                    "Average Revenue", f"${row['Avg_Revenue']:,.2f}",
                    "Revenue Lift", f"+{row['Revenue_Lift']*100:.2f}%"
                )), width=4),
            dbc.Col(stat_info, width=6) if stat_info else html.Div()
        ]), dropdown2_style  # Assuming dropdown2_style is meant for the Row


        #print(f"\nRevenue Impact:")
#print(f"A (No Bundle): ${a_rev:,.0f}")
#print(f"B (Bundle Offered): ${b_rev:,.0f}")
#print(f"Revenue lift: {lift:.2f}%")

    elif tab == 'tab_table':
        return dash_table.DataTable(
            data=filtered_data.to_dict('records'),
            columns=[{"name": col, "id": col} for col in filtered_data.columns],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "5px"},
            style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
        ), dropdown2_style

    elif tab == 'tab_ab_revenue':
        if selected_category == "All":
            return html.Div("Select a specific category to view A/B revenue comparison."), dropdown2_style
        grouped_data, p_value = test_discount_strategy(data, selected_category)
        fig = px.bar(grouped_data, x="Test Group", y="Total_Revenue", 
                     color="Test Group",
                     title=f"A/B Revenue Comparison - {selected_category}",
                     labels={"Total_Revenue": "Total Revenue ($)"})
        p_value_text = f"P-value (t-test): {p_value:.4f}"
        significance_text = "Significant difference" if p_value < 0.05 else "No significant difference"

        return html.Div([
            html.Div(f"{p_value_text} — {significance_text}", style={"marginTop": 5, "fontWeight": "bold"}),
            dcc.Graph(figure=fig)
        ]), dropdown2_style

    elif tab == 'tab_price_optimization': 
        if len(filtered_data) < 5:
            return html.Div("Not enough data for price optimization."), dropdown2_style

        avg_price = filtered_data["Price per Unit"].mean()
        avg_quantity = filtered_data["Quantity"].mean()

        X = filtered_data[["Price per Unit"]]
        y = filtered_data["Quantity"]

        model = LinearRegression().fit(X, y)
        elasticity = model.coef_[0] * (avg_price / avg_quantity)
        if elasticity > 0:
            elasticity *= -1
        prices = np.arange(avg_price * 0.5, avg_price * 2.0, 0.1)
        quantities = avg_quantity * (prices / avg_price) ** elasticity
        revenues = prices * quantities

        fig = px.line(x=prices, y=revenues, labels={"x": "Price", "y": "Revenue"},
                      title=f"Price Optimization Curve - {selected_category}")
        fig.add_scatter(x=[prices[np.argmax(revenues)]], y=[max(revenues)],
                        mode="markers", marker=dict(color="red", size=10),
                        name="Optimal Price")
        return dcc.Graph(figure=fig), dropdown2_style

    elif tab == 'tab_segmentation':
        filtered_data = filtered_data.copy()
        filtered_data["Age Group"] = pd.cut(
            filtered_data["Age"], bins=[0, 30, 50, 100], labels=["<30", "30-50", ">50"], right=False)

        age_group = filtered_data.groupby("Age Group", observed=True)["Total Amount"].mean().reset_index()
        age_fig = px.bar(age_group, x="Age Group", y="Total Amount",
                         color="Age Group",
                         title=f"Average Revenue by Age Group - {selected_category}",
                         labels={"Total Amount": "Avg Revenue"},
                         color_discrete_map={"<30": "pink", "30-50": "green", ">50" : "red"})

        gender_group = filtered_data.groupby("Gender", observed=True)["Total Amount"].mean().reset_index()
        gender_fig = px.bar(gender_group, x="Gender", y="Total Amount",
                            color = "Gender",
                            title=f"Average Revenue by Gender - {selected_category}",
                            labels={"Total Amount": "Avg Revenue"},
                            color_discrete_map = {"Total Amount": "yellow", "Avg Revenue" : "purple"}
                           )
        return [
            dcc.Graph(figure=age_fig),
            dcc.Graph(figure=gender_fig)
        ], dropdown2_style

    elif tab == 'tab_visualize':
        fig1 = px.pie(data_frame=filtered_data[filtered_data['Product Category'] != 'Bundle'], 
                     names="Product Category", values="Total Amount", hole=.3,
                     title="Total Revenue Across Categories", labels={"Total Amount": "Revenue ($)"})

        fig2 = px.box(data_frame=filtered_data[filtered_data['Product Category'] != 'Bundle'], 
                     x="Product Category", y="Total Amount",
                     color="Product Category",
                     title="Revenue Distribution by Category", labels={"Total Amount": "Revenue ($)"})

        # Filter results_df and plot_data based on selected category
        vis_results_df = results_df if selected_category == "All" else results_df[results_df["Category"] == selected_category]
        vis_plot_data = plot_data if selected_category == "All" else plot_data[plot_data["Category"] == selected_category]

        fig3 = px.bar(vis_results_df, x="Category", y="Average Sales",
                      color="Test Group", barmode="group",
                      title="Average Sales by Category and Test Group (with 10% Discount for Group B)",
                      labels={"Average Sales": "Average Sales ($)", "Category": "Product Category"},
                      color_discrete_map={"A (No Discount)": "blue", "B (Discount)": "green"})
        fig3.update_layout(template="plotly_white")

        fig4 = px.bar(vis_plot_data, x="Category", y="Amount", facet_col="Metric",
                      color="Test Group", barmode="group",
                      title="Total Profit vs Total Revenue by Category (with 10% Discount for Group B)",
                      labels={"Amount": "Amount ($)", "Category": "Product Category"},
                      color_discrete_map={"A (No Discount)": "pink", "B (Discount)": "green"})
        fig4.update_layout(template="plotly_white")

        return [
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
            dcc.Graph(figure=fig3),
            dcc.Graph(figure=fig4)
        ], dropdown2_style

@app.callback(
    Output("fig5-container", "children"),
    Input("category-dropdown-2", "value"),
    Input("tabs", "value")
)
def update_fig5(selected_category_2, tab):
    if tab != "tab_visualize":
        return None

    if selected_category_2 == "All" or not selected_category_2:

        # Create DataFrame for adoption metrics
        adoption_df = pd.DataFrame({
            'Metric': ['Total Customers', 'Bundle Adopters'],
            'Count': [len(eligible_customers), len(bundle_adopters)],
            'Type': ['Total', 'Adopted']
        })

        # Adoption metrics bar chart
        adoption_fig = px.bar(
            adoption_df,
            x='Metric',
            y='Count',
            color='Type',
            title=f"Bundle Adoption (Rate: {len(bundle_adopters)/len(eligible_customers):.1%})",
            labels={'Count': 'Number of Customers'},
            color_discrete_map={'Total': 'blue', 'Adopted': 'green'}
        )
        adoption_fig.update_layout(showlegend=False)

        # Revenue comparison bar chart
        revenue_df = pd.DataFrame({
            'Group': ['A (No Bundle)', 'B (Bundle Offered)'],
            'Revenue': [a_rev, b_rev],
            'Lift': [0, ((b_rev - a_rev)/a_rev)*100]
        })

        revenue_fig = px.bar(
            revenue_df,
            x='Group',
            y='Revenue',
            color='Group',
            title=f"Revenue Comparison (Lift: {((b_rev - a_rev)/a_rev)*100:.1f}%)",
            labels={'Revenue': 'Revenue ($)'},
            color_discrete_map={'A (No Bundle)': 'red', 'B (Bundle Offered)': 'green'}
        )
        revenue_fig.update_layout(showlegend=False)

        # Add lift annotation
        revenue_fig.add_annotation(
            x=1,  # B group position
            y=b_rev,
            text=f"+{((b_rev - a_rev)/a_rev)*100:.1f}%",
            showarrow=True,
            arrowhead=1
        )

        # All bundle prices 
        all_bundles = list(bundle_prices.items())[:15]
        bundle_data = []
        for bundle, prices in all_bundles:
            bundle_data.append({
                'Bundle': f"{bundle[0]} + {bundle[1]}",
                'Price Type': 'Regular',
                'Price': prices[0]
            })
            bundle_data.append({
                'Bundle': f"{bundle[0]} + {bundle[1]}",
                'Price Type': 'Discounted',
                'Price': prices[1]
            })
        bundle_df = pd.DataFrame(bundle_data)

        bundle_fig = px.bar(
            bundle_df,
            x='Bundle',
            y='Price',
            color='Price Type',
            barmode='group',
            title='All Bundle Pricing',
            labels={'Price': 'Price ($)'},
            color_discrete_map={'Regular': 'blue', 'Discounted': 'green'}
        )

        # Revenue Comparison
        #total_rev_data = grouped.set_index("Test Group")["Total_Revenue"]

        total_rev_fig = px.bar(
            grouped, 
            x = 'Test Group',
            y = 'Avg_Revenue',
            color = 'Test Group',
            title="Total Revenue by Group",
            color_discrete_map={'Original Price' : 'blue', '10% Price Increase' : 'orange'}
        )
        total_rev_fig.update_layout(bargap = 0.2, showlegend=False)

        #avg_qnty_data = grouped.set_index("Test Group")["Avg_Quantity"]

        avg_qnty_fig = px.bar(
            grouped, 
            x = 'Test Group',
            y = 'Avg_Quantity',
            color = 'Test Group',
            title="Average Quantity Purchased",
            color_discrete_map={'Original Price' : 'blue', '10% Price Increase' : 'orange'}
        )
        avg_qnty_fig.update_layout(bargap = 0.2, showlegend=False) #font=dict(
            #family="Times New Roman, serif",size=20,color="black"))

        return [
            html.H5("Bundle Experiment Summary"),
            html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(dcc.Graph(figure=adoption_fig), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=revenue_fig), style={'width': '50%'})],
            style={'display': 'flex', 'justifyContent': 'space-around', 'alignItems': 'flex-start'})]
        ),
            dcc.Graph(figure=bundle_fig),
            html.Div(
            children=[
                html.H5("Revenue Comparison After 10% Increase"),
                html.Div(
                    children=[
                        html.Div(dcc.Graph(figure=total_rev_fig), style={'width': '48%'}),
                        html.Div(dcc.Graph(figure=avg_qnty_fig), style={'width': '48%'})
            ],
            style={'display': 'flex', 'justifyContent': 'space-around', 'alignItems': 'flex-start'})
            ]
        )
        ]


    else:
        # Parse the selected bundle pair
        bundle_categories = selected_category_2.split('+')
        if len(bundle_categories) == 2:
            cat1, cat2 = bundle_categories
            bundle_key = (cat1, cat2)

            # Check if bundle exists in both orientations
            if bundle_key not in bundle_prices:
                bundle_key = (cat2, cat1)

            if bundle_key in bundle_prices:
                regular_price, discounted_price = bundle_prices[bundle_key]
                savings = regular_price - discounted_price
                discount_percent = (savings / regular_price) * 100

                # Create specific bundle pricing chart
                specific_bundle_data = pd.DataFrame({
                    'Bundle': [f"{cat1} + {cat2}"],
                    'Regular Price': [regular_price],
                    'Discounted Price': [discounted_price],
                    'Savings': [savings]
                })

                # Bundle pricing comparison for selected pair
                price_comparison_data = []
                price_comparison_data.append({
                    'Bundle': f"{cat1} + {cat2}",
                    'Price Type': 'Regular',
                    'Price': regular_price
                })
                price_comparison_data.append({
                    'Bundle': f"{cat1} + {cat2}",
                    'Price Type': 'Discounted',
                    'Price': discounted_price
                })

                price_comparison_df = pd.DataFrame(price_comparison_data)

                bundle_fig_specific = px.bar(
                    price_comparison_df,
                    x='Bundle',
                    y='Price',
                    color='Price Type',
                    barmode='group',
                    title=f'Bundle Pricing: {cat1} + {cat2}',
                    labels={'Price': 'Price ($)'},
                    color_discrete_map={'Regular': 'blue', 'Discounted': 'green'}
                )

                # Add savings annotation
                bundle_fig_specific.add_annotation(
                    x=0,
                    y=regular_price,
                    text=f"Save ${savings:.2f}<br>({discount_percent:.1f}% off)",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor="red",
                    arrowwidth=2
                )

                # Get individual category prices for comparison
                avg_unit_price = data.groupby('Product Category')['Price per Unit'].mean()
                individual_prices_data = []
                individual_prices_data.append({
                    'Category': cat1,
                    'Type': 'Individual',
                    'Price': avg_unit_price[cat1]
                })
                individual_prices_data.append({
                    'Category': cat2,
                    'Type': 'Individual', 
                    'Price': avg_unit_price[cat2]
                })
                individual_prices_data.append({
                    'Category': f'{cat1}+{cat2} Bundle',
                    'Type': 'Bundle',
                    'Price': discounted_price
                })

                individual_vs_bundle_df = pd.DataFrame(individual_prices_data)

                comparison_fig = px.bar(
                    individual_vs_bundle_df,
                    x='Category',
                    y='Price',
                    color='Type',
                    title=f'Individual vs Bundle Pricing: {cat1} + {cat2}',
                    labels={'Price': 'Price ($)'},
                    color_discrete_map={'Individual': 'orange', 'Bundle': 'green'}
                )

                return [
                    html.H3(f"Selected Bundle: {cat1} + {cat2}"),
                    html.P(f"Regular Price: ${regular_price:.2f}"),
                    html.P(f"Bundle Price: ${discounted_price:.2f}"),
                    html.P(f"You Save: ${savings:.2f} ({discount_percent:.1f}% discount)"),
                    dcc.Graph(figure=bundle_fig_specific),
                    dcc.Graph(figure=comparison_fig)
                ]
            else:
                return html.Div(f"Bundle pricing not available for {selected_category_2}")
        else:
            return html.Div("Invalid bundle selection")

# Run app
if __name__ == "__main__":
    app.run(debug=True, port=8052)
