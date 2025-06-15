# Pandas Cheatsheet - Complete Gids met Voorbeelden

## 📊 Basis Concepten

### Wat is Pandas?
Pandas is een krachtige Python library voor data-analyse, gebouwd bovenop NumPy. Het biedt twee hoofdstructuren:
- **Series**: 1D gelabelde array (zoals een kolom in Excel)
- **DataFrame**: 2D tabel met rijen en kolommen (zoals een Excel spreadsheet)

### Waarom Pandas vs NumPy?
```python
# NumPy: alleen impliciete (positie) indexing
import numpy as np
np_array = np.array([25, 30, 35])
# Toegang: np_array[0] = 25

# Pandas: expliciete (label) + impliciete indexing
import pandas as pd
ages = pd.Series([25, 30, 35], index=['Alice', 'Bob', 'Charlie'])
# Toegang: ages['Alice'] = 25 OF ages[0] = 25
```
---

## 🏗️ Data Structures

### Series Maken
```python
# Van lijst met expliciete index
names = pd.Series(['Alice', 'Bob', 'Charlie'], index=[1, 2, 3])

# Van dictionary (keys worden index)
ages = pd.Series({'Alice': 25, 'Bob': 30, 'Charlie': 35})

# Van NumPy array
scores = pd.Series(np.random.randn(5))
```

### DataFrame Maken
```python
# Van dictionary
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['Brussels', 'Antwerp', 'Ghent']
})

# Van lijst met dictionaries
data = [
    {'Name': 'Alice', 'Age': 25, 'City': 'Brussels'},
    {'Name': 'Bob', 'Age': 30, 'City': 'Antwerp'}
]
df = pd.DataFrame(data)

# Van NumPy array met expliciete labels
df = pd.DataFrame(
    np.random.randn(4, 3),
    columns=['A', 'B', 'C'],
    index=['row1', 'row2', 'row3', 'row4']
)
```

---

## 📥 Data Inladen

### CSV & Excel
```python
# CSV inladen
df = pd.read_csv('airlines.csv')
df = pd.read_csv('file.csv', sep=';', decimal=',')  # EU format
df = pd.read_csv('file.csv', encoding='utf-8')     # Character encoding

# Excel inladen
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_excel('data.xlsx', sheet_name=0)      # Eerste sheet
```

### Data Types bij Import
```python
# Specifieke datatypes opgeven
df = pd.read_csv('airlines.csv', dtype={
    'Gender': 'category',
    'Age': 'int32',
    'Nationality': 'category'
})

# Datums automatisch parsen
df = pd.read_csv('file.csv', parse_dates=['date_column'])
```

---

## 🔍 Data Exploratie

### Eerste Indruk
```python
# Dataset overzicht
df.info()              # Kolommen, datatypes, memory usage
df.describe()          # Statistieken voor numerieke kolommen
df.describe(include='all')  # Alle kolommen inclusief categorische

# Vorm en structuur
df.shape              # (aantal_rijen, aantal_kolommen)
df.head(10)           # Eerste 10 rijen
df.tail(5)            # Laatste 5 rijen
df.sample(3)          # 3 willekeurige rijen
```

### Kolom Informatie
```python
# Kolomnamen en types
df.columns.tolist()    # Lijst van kolomnamen
df.dtypes             # Datatype per kolom
df.index              # Index informatie

# Unieke waarden
df['Gender'].unique()                    # Alle unieke waarden
df['Gender'].nunique()                   # Aantal unieke waarden
df['Gender'].value_counts()              # Frequentie van elke waarde
df['Gender'].value_counts(normalize=True) # Relatieve frequentie
```

---

## 🎯 Indexing & Selection - Complete Gids

### 🔑 Fundamentele Concepten

**Elke Pandas object heeft TWEE indexen:**
1. **Expliciete index**: De labels die je zelf definieert
2. **Impliciete index**: De posities (0, 1, 2, 3, ...)

```python
# Voorbeeld met je airlines data
df = pd.read_csv('airlines.csv')
print(df.index)        # RangeIndex(0, 37) - impliciete index
print(df.columns)      # Expliciete labels voor kolommen

# Custom index maken
df_custom = df.set_index('Name')
print(df_custom.index) # Namen als expliciete index
```

### 📍 .loc vs .iloc - Het Verschil

#### .loc = **L**abel-based (Expliciete Index)
```python
# Airlines dataset voorbeelden
df = pd.read_csv('airlines.csv')

# Enkele rij op index positie (let op: dit is verwarrend!)
df.loc[0]              # Rij met index label 0

# Specifieke cel
df.loc[0, 'Name']      # Eerste persoon naam
df.loc[5, 'Age']       # Leeftijd van persoon op index 5

# Meerdere rijen
df.loc[0:5]            # Rijen 0 t/m 5 (INCLUSIEF 5!)
df.loc[[0, 2, 4]]      # Specifieke rijen 0, 2, en 4

# Kolom selectie
df.loc[:, 'Name']              # Alle rijen, kolom Name
df.loc[:, 'Name':'Age']        # Alle rijen, kolommen Name t/m Age
df.loc[:, ['Name', 'Gender', 'Nationality']]  # Specifieke kolommen

# 2D selectie (rijen EN kolommen)
df.loc[0:5, 'Name':'Age']      # Eerste 6 rijen, kolommen Name t/m Age
df.loc[[0, 2, 4], ['Name', 'Gender']]  # Specifieke rijen en kolommen
```

#### .iloc = **I**nteger positions (Impliciete Index)
```python
# Position-based indexing
df.iloc[0]             # Eerste rij (positie 0)
df.iloc[-1]            # Laatste rij
df.iloc[0, 1]          # Eerste rij, tweede kolom

# Slicing (exclusief eindpunt!)
df.iloc[0:5]           # Eerste 5 rijen (0, 1, 2, 3, 4)
df.iloc[:, 0:3]        # Alle rijen, eerste 3 kolommen
df.iloc[0:5, 1:4]      # Eerste 5 rijen, kolommen 1-3

# Lijst van posities
df.iloc[[0, 5, 10]]           # Rijen op posities 0, 5, 10
df.iloc[:, [0, 2, 4]]         # Kolommen op posities 0, 2, 4

# Negatieve indexing
df.iloc[-5:]           # Laatste 5 rijen
df.iloc[:, -2:]        # Laatste 2 kolommen
```

### 🎨 Kleur Selectie Voorbeelden (Gebaseerd op Airlines Data)

```python
# Stel we hebben een dataset met kleuren
colors_df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Favorite_Color': ['Red', 'Blue', 'Green', 'Yellow', 'Purple'],
    'Age': [25, 30, 35, 28, 32],
    'Country': ['Belgium', 'Netherlands', 'France', 'Germany', 'Spain']
})

# EXPLICIETE INDEXING met .loc
# Alle rode en blauwe kleuren
red_blue_mask = colors_df['Favorite_Color'].isin(['Red', 'Blue'])
colors_df.loc[red_blue_mask]

# Mensen ouder dan 30 met hun kleuren
old_people = colors_df['Age'] > 30
colors_df.loc[old_people, ['Name', 'Favorite_Color']]

# IMPLICIETE INDEXING met .iloc
# Eerste 3 mensen en hun kleuren (posities 0, 1, 2)
colors_df.iloc[0:3, [0, 1]]  # Name en Favorite_Color kolommen

# Elke tweede persoon
colors_df.iloc[::2, :]       # Posities 0, 2, 4

# Laatste 2 mensen met alleen naam en kleur
colors_df.iloc[-2:, [0, 1]]

# Airlines dataset - Gender selectie voorbeelden
# Alle vrouwen
female_mask = df['Gender'] == 'Female'
df.loc[female_mask, ['Name', 'Age', 'Nationality']]

# Mannen uit China
chinese_men = (df['Gender'] == 'Male') & (df['Nationality'] == 'China')
df.loc[chinese_men]

# Eerste 10 rijen, alleen persoonlijke info
df.iloc[:10, :4]  # Eerste 4 kolommen (Name, Gender, Age, Nationality)
```

### 🔍 Boolean Indexing - Geavanceerd

```python
# Airlines dataset voorbeelden
df = pd.read_csv('airlines.csv')

# Basis filtering
young_people = df[df['Age'] < 30]
european_airports = df[df['Airport Continent'] == 'EU']

# Meerdere condities (BELANGRIJK: gebruik & en | niet 'and'/'or')
young_women = df[(df['Age'] < 30) & (df['Gender'] == 'Female')]
europe_asia = df[(df['Airport Continent'] == 'EU') | (df['Airport Continent'] == 'AS')]

# isin() voor meerdere waarden
asian_countries = df[df['Nationality'].isin(['China', 'Japan', 'Vietnam', 'Thailand'])]
north_american_airports = df[df['Airport Country Code'].isin(['US', 'CA'])]

# String operaties
us_airports = df[df['Airport Name'].str.contains('Airport')]
names_with_a = df[df['Name'].str.startswith('a')]

# Inverse filtering met ~
not_female = df[~(df['Gender'] == 'Female')]  # Equivalent aan df['Gender'] != 'Female'
not_asian = df[~df['Nationality'].isin(['China', 'Japan', 'Vietnam'])]
```

### 🚀 Query Method - SQL-achtige Syntax

```python
# Readable filtering met query()
df.query('Age > 30')
df.query('Age > 30 and Gender == "Female"')
df.query('Nationality == "China" or Nationality == "Japan"')

# Met variabelen
min_age = 25
df.query('Age >= @min_age')

# String operaties in query
df.query('Name.str.len() > 5')
df.query('Airport_Name.str.contains("International")')

# Complex conditions
df.query('Age > 30 and (Nationality == "China" or Nationality == "Japan")')
```

---

## 🏷️ Kolom Operaties

### Kolom Selectie
```python
# Enkele kolom (geeft Series)
names = df['Name']
ages = df['Age']

# Meerdere kolommen (geeft DataFrame)
personal_info = df[['Name', 'Gender', 'Age']]
airport_info = df[['Airport Name', 'Airport Country Code', 'Country Name']]

# Kolom slicing (als kolommen gesorteerd zijn)
df.loc[:, 'Name':'Nationality']  # Van Name tot Nationality

# Alle kolommen behalve enkele
df.drop(['Airport Name', 'Airport Country Code'], axis=1)
df.loc[:, ~df.columns.isin(['Name', 'Gender'])]
```

### Kolommen Toevoegen & Wijzigen
```python
# Nieuwe kolom toevoegen
df['Age_Group'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Old')
df['Full_Info'] = df['Name'] + ' (' + df['Gender'] + ')'

# Kolom op specifieke positie invoegen
df.insert(1, 'ID', range(1, len(df) + 1))

# Kolom hernoemen
df.rename(columns={'Name': 'Passenger_Name', 'Age': 'Passenger_Age'}, inplace=True)

# Kolom verwijderen
df.drop('Airport Name', axis=1, inplace=True)  # Permanent
df_without_age = df.drop('Age', axis=1)        # Nieuwe DataFrame
```

---

## 🧹 Missing Values - Praktische Aanpak

### Detecteren
```python
# Missing values checken
df.isnull().sum()           # Aantal missing per kolom
df.isnull().any()           # Kolommen met missing values
df.isnull().sum().sum()     # Totaal aantal missing values

# Percentage missing
(df.isnull().sum() / len(df) * 100).round(2)

# Rijen met missing values
rows_with_missing = df[df.isnull().any(axis=1)]
```

### Behandelen
```python
# Verwijderen
df_clean = df.dropna()                    # Alle rijen met NaN weg
df_clean = df.dropna(subset=['Age'])      # Alleen rijen met NaN in Age
df_clean = df.dropna(thresh=5)            # Behoud rijen met min 5 non-NaN

# Opvullen
df['Age'].fillna(df['Age'].mean(), inplace=True)        # Met gemiddelde
df['Gender'].fillna('Unknown', inplace=True)            # Met constante waarde
df.fillna(method='ffill', inplace=True)                 # Forward fill
df.fillna(method='bfill', inplace=True)                 # Backward fill

# Interpolatie voor numerieke data
df['Age'].interpolate(inplace=True)
```

---

## 🔢 String Operaties - Vectorized Power

### Basis Transformaties
```python
# Case conversions
df['Name'].str.lower()           # naar lowercase
df['Name'].str.upper()           # naar uppercase
df['Name'].str.title()           # Title Case
df['Name'].str.capitalize()      # Eerste letter hoofdletter

# String eigenschappen
df['Name'].str.len()             # Lengte van elke naam
df['Airport Name'].str.count('Airport')  # Tel woord 'Airport'
df['Name'].str.startswith('A')   # Begint met 'A'
df['Airport Name'].str.endswith('Airport')  # Eindigt met 'Airport'
```

### String Manipulatie
```python
# Slicing
df['Name'].str[0]                # Eerste karakter
df['Name'].str[:3]               # Eerste 3 karakters
df['Name'].str[-1]               # Laatste karakter
df['Name'].str[1:4]              # Karakters 1 t/m 3

# Cleaning
df['Name'].str.strip()           # Whitespace weg
df['Airport Name'].str.replace('Airport', 'Airfield')  # Vervangen

# Splitten
df['Name'].str.split(' ')                    # Split op spatie
df['Name'].str.split(' ', expand=True)       # Split naar kolommen
name_parts = df['Name'].str.split(' ', n=1, expand=True)
df['First_Name'] = name_parts[0]
df['Last_Name'] = name_parts[1]
```

### Pattern Matching
```python
# Contains
us_airports = df[df['Airport Name'].str.contains('International')]
chinese_names = df[df['Name'].str.contains('ng')]

# Regex operaties
phone_pattern = r'\d{3}-\d{3}-\d{4}'
df['Phone'].str.match(phone_pattern)         # Match vanaf begin
df['Phone'].str.findall(phone_pattern)       # Alle matches
df['Phone'].str.extract(r'(\d{3})-(\d{3})-(\d{4})')  # Groepen extraheren
```

---

## 📊 GroupBy - Split-Apply-Combine

### Basis GroupBy
```python
# Airlines dataset voorbeelden
# Groeperen op één kolom
by_gender = df.groupby('Gender')
by_continent = df.groupby('Airport Continent')

# Basis aggregaties
df.groupby('Gender')['Age'].mean()           # Gemiddelde leeftijd per gender
df.groupby('Gender').size()                  # Aantal personen per gender
df.groupby('Airport Continent').count()      # Count per continent
```

### Geavanceerde Aggregaties
```python
# Meerdere aggregaties tegelijk
df.groupby('Gender')['Age'].agg(['min', 'max', 'mean', 'std'])

# Verschillende aggregaties per kolom
df.groupby('Gender').agg({
    'Age': ['mean', 'std'],
    'Name': 'count',
    'Nationality': 'nunique'
})

# Custom aggregatie functies
def age_range(ages):
    return ages.max() - ages.min()

df.groupby('Gender')['Age'].agg(age_range)

# Groeperen op meerdere kolommen
df.groupby(['Gender', 'Airport Continent'])['Age'].mean()
```

### GroupBy Transformaties
```python
# Transform (behoudt originele vorm)
df['Age_Normalized'] = df.groupby('Gender')['Age'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Apply custom functies
def top_ages(group):
    return group.nlargest(2, 'Age')

df.groupby('Gender').apply(top_ages)

# Filter groepen
df.groupby('Nationality').filter(lambda x: len(x) >= 2)  # Landen met 2+ personen
```

---

## 🔗 Merge & Join - Data Combineren

### Merge Operaties
```python
# Stel we hebben twee datasets
passengers = df[['Name', 'Gender', 'Age']].head(5)
flights = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Flight_Number': ['AA123', 'BB456', 'CC789', 'DD012', 'EE345'],
    'Departure': ['Brussels', 'Amsterdam', 'Paris', 'Berlin', 'Madrid']
})

# Basic merge (inner join op gemeenschappelijke kolommen)
combined = pd.merge(passengers, flights, on='Name')

# Verschillende join types
pd.merge(passengers, flights, on='Name', how='inner')    # Alleen matches
pd.merge(passengers, flights, on='Name', how='outer')    # Alle rijen
pd.merge(passengers, flights, on='Name', how='left')     # Alle uit passengers
pd.merge(passengers, flights, on='Name', how='right')    # Alle uit flights

# Verschillende kolomnamen
pd.merge(passengers, flights, left_on='Name', right_on='Passenger_Name')

# Overlappende kolomnamen
pd.merge(df1, df2, on='key', suffixes=['_left', '_right'])
```

### Join Operations
```python
# Join (standaard left join op index)
df1.join(df2)                    # Left join op index
df1.join(df2, how='outer')       # Outer join
df1.join(df2, rsuffix='_right')  # Suffix voor overlappende kolommen

# Concatenate
pd.concat([df1, df2])                        # Rijen onder elkaar
pd.concat([df1, df2], axis=1)                # Kolommen naast elkaar
pd.concat([df1, df2], ignore_index=True)     # Nieuwe index
```

---

## 📈 Data Transformatie & Analyse

### Sorting
```python
# Sorteren op één kolom
df.sort_values('Age')                        # Oplopend
df.sort_values('Age', ascending=False)       # Aflopend

# Sorteren op meerdere kolommen
df.sort_values(['Gender', 'Age'])            # Eerst Gender, dan Age
df.sort_values(['Gender', 'Age'], ascending=[True, False])  # Mixed

# Sorteren op index
df.sort_index()                              # Index oplopend
```

### Apply & Map
```python
# Apply op kolommen
df['Age_Category'] = df['Age'].apply(
    lambda x: 'Young' if x < 30 else 'Middle' if x < 50 else 'Senior'
)

# Apply op hele DataFrame
df.apply(lambda row: f"{row['Name']} is {row['Age']} years old", axis=1)

# Map voor simpele vervanging
gender_map = {'Male': 'M', 'Female': 'F'}
df['Gender_Short'] = df['Gender'].map(gender_map)
```

### Pivot Tables
```python
# Basic pivot
pivot = df.pivot_table(
    values='Age', 
    index='Gender', 
    columns='Airport Continent', 
    aggfunc='mean'
)

# Meerdere waarden
pivot_multi = df.pivot_table(
    values=['Age'], 
    index=['Gender'], 
    columns=['Airport Continent'],
    aggfunc=['mean', 'count']
)
```

---

## 🎯 Praktische Voorbeelden - Airlines Dataset

### Scenario 1: Demografische Analyse
```python
# Leeftijdsverdeling per gender
age_stats = df.groupby('Gender')['Age'].describe()

# Top 5 nationaliteiten
top_nationalities = df['Nationality'].value_counts().head()

# Gemiddelde leeftijd per continent
continent_ages = df.groupby('Airport Continent')['Age'].mean().sort_values(ascending=False)
```

### Scenario 2: Luchthaven Analyse
```python
# Welke landen hebben de meeste luchthavens in de dataset?
airport_countries = df['Country Name'].value_counts()

# Passagiers per continent
passengers_per_continent = df.groupby('Airport Continent').size()

# Oudste passagier per continent
oldest_per_continent = df.groupby('Airport Continent')['Age'].max()
```

### Scenario 3: Filtering & Selection
```python
# Alle Aziatische vrouwen onder 30
young_asian_women = df[
    (df['Gender'] == 'Female') & 
    (df['Age'] < 30) & 
    (df['Airport Continent'] == 'AS')
]

# Passagiers naar Noord-Amerikaanse luchthavens
north_american_passengers = df[df['Airport Continent'] == 'NAM']

# Luchthavens met 'International' in de naam
international_airports = df[df['Airport Name'].str.contains('International', na=False)]
```

---

## ⚡ Performance Tips

### Memory Optimization
```python
# Categoricals voor herhaalde strings
df['Gender'] = df['Gender'].astype('category')
df['Nationality'] = df['Nationality'].astype('category')

# Kleinere numeric types
df['Age'] = df['Age'].astype('int8')  # Als alle leeftijden < 255

# Memory usage checken
df.info(memory_usage='deep')
df.memory_usage(deep=True)
```

### Efficient Operations
```python
# Vectorized operaties zijn sneller dan loops
# SLECHT:
# for i in range(len(df)):
#     df.loc[i, 'Age_Double'] = df.loc[i, 'Age'] * 2

# GOED:
df['Age_Double'] = df['Age'] * 2

# Method chaining voor leesbaarheid
result = (df
    .query('Age > 25')
    .groupby('Gender')
    .agg({'Age': 'mean'})
    .reset_index()
    .sort_values('Age', ascending=False)
)
```

---

## ⚠️ Veelgemaakte Fouten

### 1. Chained Indexing
```python
# ❌ FOUT: Kan SettingWithCopyWarning geven
# df['Age'][df['Gender'] == 'Female'] = df['Age'] + 1

# ✅ GOED: Gebruik .loc
df.loc[df['Gender'] == 'Female', 'Age'] = df['Age'] + 1
```

### 2. Boolean Logic
```python
# ❌ FOUT: Gebruik van 'and'/'or'
# df[(df['Age'] > 25) and (df['Gender'] == 'Female')]

# ✅ GOED: Gebruik '&'/'|' met parentheses
df[(df['Age'] > 25) & (df['Gender'] == 'Female')]
```

### 3. Index Confusion
```python
# ❌ VERWARREND: Direct indexing kan misleiden
# df[0]  # Dit geeft fout! df[0] probeert kolom '0' te vinden

# ✅ DUIDELIJK: Gebruik .loc/.iloc
df.iloc[0]    # Eerste rij
df.loc[0]     # Rij met index label 0
```

### 4. Copy vs View
```python
# ✅ VEILIG: Expliciete copy maken
df_subset = df[df['Age'] > 30].copy()
df_subset.loc[:, 'New_Column'] = 'Value'  # Geen warning
```

---

## 🔧 Debugging & Troubleshooting

### Data Inspection
```python
# Quick checks
df.head()
df.info()
df.describe()
df.sample(5)

# Data quality checks
df.duplicated().sum()           # Aantal duplicaten
df.isnull().sum()              # Missing values
df.dtypes                      # Data types
```

### Performance Monitoring
```python
# Memory usage
df.memory_usage(deep=True).sum()

# Timing operations (in Jupyter)
# %timeit df.groupby('Gender')['Age'].mean()

# Profile specific operations
import time
start = time.time()
result = df.groupby('Gender')['Age'].mean()
print(f"Operation took {time.time() - start:.4f} seconds")
```

---

## 📚 Handige Shortcuts

### Quick Data Exploration
```python
# One-liner dataset overview
def quick_overview(df):
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

quick_overview(df)
```

### Common Patterns
```python
# Top N per group
df.groupby('Gender').apply(lambda x: x.nlargest(3, 'Age'))

# Percentage of total
df.groupby('Gender').size() / len(df) * 100

# Conditional column creation
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 50, 100], labels=['Young', 'Middle', 'Senior'])

# Multiple condition filtering
mask = (
    (df['Age'] > 25) & 
    (df['Gender'] == 'Female') & 
    (df['Airport Continent'].isin(['EU', 'AS']))
)
filtered_df = df[mask]
```

Dit is een complete gids die je kunt gebruiken als referentie voor al je Pandas operaties!