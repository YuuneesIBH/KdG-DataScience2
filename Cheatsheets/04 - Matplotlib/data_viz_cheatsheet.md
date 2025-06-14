# Data Visualization Cheatsheet

## Setup
```python
# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

## Line Chart
**When to use:** Time series, trends over time, continuous monitoring
```python
# Basic line chart
plt.plot(x, y, marker='o', label='Line 1')
plt.title('Title')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
```

## Scatter Plot
**When to use:** Relationships/correlations between two variables
```python
# Basic scatter plot
plt.scatter(x, y, color='green', marker='o')
plt.title('Scatter Plot')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.show()
```

## Histogram
**When to use:** Distribution of continuous variables, interval/ratio data
```python
# Basic histogram
plt.hist(data, bins=20, density=True, color='purple', 
         alpha=0.3, edgecolor='white')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

## Bar Chart (Seaborn)
**When to use:** Comparing discrete categories, nominal/ordinal data
```python
# Basic bar chart
sns.barplot(data=df, x='category', y='value')
plt.title('Bar Chart')
plt.show()

# Stacked bar chart
sns.barplot(data=df, x='category', y='value', hue='subcategory')
```

## Box Plot (Seaborn)
**When to use:** Visualizing dispersion, outliers, comparing datasets
```python
# Basic boxplot
sns.boxplot(data=df, x='category', y='value')
plt.title('Box Plot')
plt.show()

# Grouped boxplot
sns.boxplot(data=df, x='category', y='value', hue='group')
```

## Pie Chart
**When to use:** Parts of whole (100%), max 5 categories
```python
# Basic pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()
```

## Quick Reference - When to Use What

| Chart Type | Use For | Data Type |
|------------|---------|-----------|
| **Line Chart** | Trends over time, time series | Continuous |
| **Scatter Plot** | Relationships between variables | Continuous |
| **Histogram** | Data distribution | Continuous |
| **Bar Chart** | Compare categories | Categorical |
| **Box Plot** | Show dispersion, outliers | Continuous |
| **Pie Chart** | Parts of whole (≤5 categories) | Categorical |

## Common Mistakes to Avoid
- ❌ Too many categories in pie/line charts
- ❌ Manipulating Y-axis scales
- ❌ Wrong chart type for data
- ❌ Too much information in one chart

## Best Practices
- ✅ Show only necessary information
- ✅ Use multiple simple charts instead of one complex chart
- ✅ Choose appropriate chart for data type
- ✅ Bar chart vs Pie chart: Use bar for exact differences
- ✅ Histogram vs Line chart: Histogram for distributions, line for time series