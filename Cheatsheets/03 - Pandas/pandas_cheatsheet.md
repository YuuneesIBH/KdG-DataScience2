## String Operaties - Complete Guide

### Basis String Methoden - Vectorized
```python
# Basis transformaties
df['text'].str.lower()           # Naar lowercase
df['text'].str.upper()           # Naar uppercase  
df['text'].str.title()           # Title Case
df['text'].str.capitalize()      # Eerste letter hoofdletter
df['text'].str.swapcase()        # Wissel hoofd/kleine letters

# String eigenschappen
df['text'].str.len()             # Lengte van elke string
df['text'].str.count('a')        # Tel occurrences van 'a'
df['text'].str.startswith('A')   # Boolean: begint met 'A'
df['text'].str.endswith('ing')   # Boolean: eindigt met 'ing'
df['text'].str.isalpha()         # Boolean: alleen letters
df['text'].str.isnumeric()       # Boolean: alleen nummers
df['text'].str.isalnum()         # Boolean: letters en nummers
```

### String Slicing en Indexing
```python
# Character-wise indexing
df['text'].str[0]                # Eerste karakter van elke string
df['text'].str[-1]               # Laatste karakter
df['text'].str[1:4]              # Karakters 1 tot 4 (exclusief 4)
df['text'].str[:3]               # Eerste 3 karakters
df['text'].str[2:]               # Vanaf 3e karakter tot einde

# Slicing met step
df['text'].str[::2]              # Elke tweede karakter
df['text'].str[::-1]             # String omkeren

# Get specific character positions
df['text'].str.get(0)            # Veiligere versie van str[0] (geeft NaN als index niet bestaat)
```

### String Cleaning en Formatting
```python
# Whitespace handling
df['text'].str.strip()           # Verwijder leading/trailing whitespace
df['text'].str.lstrip()          # Alleen leading whitespace
df['text'].str.rstrip()          # Alleen trailing whitespace
df['text'].str.strip('.,')       # Verwijder specifieke karakters

# Padding
df['text'].str.pad(10, side='left', fillchar='0')    # Links padding met 0
df['text'].str.pad(10, side='right', fillchar=' ')   # Rechts padding met spatie
df['text'].str.pad(10, side='both', fillchar='*')    # Centreren met *
df['text'].str.center(10, fillchar='-')              # Centreren (shorthand)
df['text'].str.ljust(10, fillchar='0')               # Links justify
df['text'].str.rjust(10, fillchar='0')               # Rechts justify
df['text'].str.zfill(10)                             # Zero-fill (voor nummers als strings)

# Case conversions
df['text'].str.lower()
df['text'].str.upper()
df['text'].# Pandas Cheatsheet - Complete Guide

## Data Inladen en Voorbereiden

### Basic Data Import
```python
import pandas as pd
import numpy as np

# CSV files
data = pd.read_csv('file.csv')
data = pd.read_csv('file.csv', sep=';')  # Andere separator
data = pd.read_csv('file.csv', sep=';', decimal=',')  # Andere decimal separator
data = pd.read_csv('file.csv', names=['col1', 'col2'])  # Geen header in file

# Excel files
data = pd.read_excel('file.xlsx')
data = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# JSON files
data = pd.read_json('file.json')
```

### Data Types en Categoricals
```python
# Categorische variabelen definiëren bij import
data = pd.read_csv('file.csv', dtype={'brand': 'category', 'type': 'category'})

# Categorische variabelen aanmaken
bloodtype = pd.Categorical(['O-', 'B-', 'B-', 'A+'], 
                          categories=['O-','O+','B-','B+','A-','A+','AB-','AB+'])

# Kolom omzetten naar categorical
data['column'] = data['column'].astype('category')
```

## Data Exploratie

### Basic Info
```python
# Overzicht van data
data.info()              # Datatypes, memory usage, non-null counts
data.describe()          # Statistische samenvatting numerieke kolommen
data.describe(include='all')  # Alle kolommen inclusief categorische

# Vorm en structuur
data.shape              # (rows, columns)
data.head()             # Eerste 5 rijen
data.tail()             # Laatste 5 rijen
data.columns            # Kolomnamen
data.index              # Index labels
data.dtypes             # Datatypes per kolom
```

### Unieke Waarden en Frequenties
```python
# Unieke waarden
data['column'].unique()
data['column'].nunique()        # Aantal unieke waarden
data['column'].value_counts()   # Frequentie van elke waarde
data['column'].value_counts(normalize=True)  # Relatieve frequentie

# Voor alle kolommen
pd.unique(data['column'])       # Numpy variant
```

## Data Structures

### Series Aanmaken
```python
# Verschillende manieren om Series aan te maken
s1 = pd.Series([1, 2, 3, 4])                    # Standaard integer index
s2 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])  # Expliciete index
s3 = pd.Series({'a': 1, 'b': 2, 'c': 3})       # Van dictionary

# Series eigenschappen
s2.values    # Numpy array van waarden
s2.index     # Index object
s2.dtype     # Datatype
```

### DataFrame Aanmaken
```python
# Van lijst met dictionaries
df = pd.DataFrame([
    {'name': 'Alice', 'age': 25, 'city': 'Brussels'},
    {'name': 'Bob', 'age': 30, 'city': 'Antwerp'}
])

# Van dictionary met Series
population = pd.Series({'BE': 11.7, 'NL': 17.7})
area = pd.Series({'BE': 30688, 'NL': 41850})
df = pd.DataFrame({'population': population, 'area': area})

# Van numpy array
df = pd.DataFrame(np.random.randn(4, 3), 
                  columns=['A', 'B', 'C'],
                  index=['row1', 'row2', 'row3', 'row4'])
```

## Indexing en Selection - Complete Guide

### Index Concepts - Het Verschil Begrijpen

**Expliciete vs Impliciete Index:**
```python
# NumPy array: alleen impliciete (positie) index
np_array = np.array([10, 20, 30, 40])
# Index: 0, 1, 2, 3 (automatisch, niet zichtbaar)

# Pandas Series: expliciete index PLUS impliciete index
pd_series = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
# Expliciete index: 'a', 'b', 'c', 'd' (zelf gedefinieerd)
# Impliciete index: 0, 1, 2, 3 (nog steeds aanwezig!)

print(pd_series.index)      # Index(['a', 'b', 'c', 'd'])
print(pd_series.values)     # array([10, 20, 30, 40])
```

### Series Indexing - Alle Methoden

#### Basic Indexing
```python
data = pd.Series(['alpha', 'beta', 'gamma', 'delta'], index=[1, 3, 5, 7])

# Directe index access (gebruikt expliciete index)
data[1]          # 'alpha' - WAARSCHUWING: kan verwarrend zijn!
data[3]          # 'beta'

# Als expliciete index integers zijn: VERWARRING MOGELIJK
series_confusing = pd.Series(['a', 'b', 'c'], index=[1, 0, 2])
series_confusing[1]    # 'a' (gebruikt expliciete index 1)
series_confusing[0]    # 'b' (gebruikt expliciete index 0)
# DAAROM: Gebruik altijd .loc en .iloc voor duidelijkheid!
```

#### .loc - Label Based Indexing (EXPLICIETE index)
```python
data = pd.Series(['alpha', 'beta', 'gamma', 'delta'], index=[1, 3, 5, 7])

# Enkele waarde
data.loc[1]         # 'alpha'
data.loc[5]         # 'gamma'

# Slicing (BELANGRIJK: inclusief eindpunt!)
data.loc[1:5]       # 1:'alpha', 3:'beta', 5:'gamma' (alle drie!)
data.loc[3:7]       # 3:'beta', 5:'gamma', 7:'delta'

# Lijst van labels
data.loc[[1, 5]]    # Labels 1 en 5
data.loc[[7, 1]]    # Labels 7 en 1 (volgorde wordt gerespecteerd)

# Boolean mask
mask = data.str.contains('a')
data.loc[mask]      # Alle waarden die 'a' bevatten

# Conditionele selectie
data.loc[data.str.len() > 4]  # Strings langer dan 4 karakters
```

#### .iloc - Position Based Indexing (IMPLICIETE index)
```python
data = pd.Series(['alpha', 'beta', 'gamma', 'delta'], index=[1, 3, 5, 7])

# Enkele positie
data.iloc[0]        # 'alpha' (eerste positie)
data.iloc[-1]       # 'delta' (laatste positie)

# Slicing (BELANGRIJK: exclusief eindpunt!)
data.iloc[1:3]      # Posities 1 en 2: 'beta', 'gamma'
data.iloc[:2]       # Eerste twee: 'alpha', 'beta'
data.iloc[2:]       # Vanaf positie 2: 'gamma', 'delta'

# Lijst van posities
data.iloc[[0, 2]]   # Posities 0 en 2
data.iloc[[3, 1, 0]]  # Posities in specifieke volgorde

# Negatieve indexing
data.iloc[-2:]      # Laatste twee elementen
```

#### Advanced Series Indexing
```python
# String index
data_str = pd.Series([1, 2, 3, 4], index=['apple', 'banana', 'cherry', 'date'])

# Partial string matching (alleen bij sorteerde index)
data_str = data_str.sort_index()
data_str.loc['a':'c']  # Van 'apple' tot alles wat begint met 'c'

# Boolean combinaties
data = pd.Series(range(10))
data.loc[(data > 3) & (data < 7)]  # Tussen 3 en 7
data.loc[(data < 2) | (data > 8)]  # Kleiner dan 2 OF groter dan 8

# isin voor membership testing
data_str.loc[data_str.index.isin(['apple', 'cherry'])]
```

### DataFrame Indexing - Complete Guide

#### Setup voorbeeld data
```python
df = pd.DataFrame({
    'naam': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'leeftijd': [25, 30, 35, 28],
    'stad': ['Brussel', 'Antwerpen', 'Gent', 'Leuven'],
    'salaris': [50000, 60000, 55000, 52000]
}, index=['emp1', 'emp2', 'emp3', 'emp4'])

print(df)
#        naam  leeftijd      stad  salaris
# emp1   Alice        25   Brussel    50000
# emp2     Bob        30  Antwerpen    60000
# emp3 Charlie        35       Gent    55000
# emp4   Diana        28     Leuven    52000
```

#### Kolom Selectie - Verschillende Methoden
```python
# Enkele kolom (geeft Series terug)
df['naam']                    # Series
type(df['naam'])              # <class 'pandas.core.series.Series'>

# Meerdere kolommen (geeft DataFrame terug)
df[['naam', 'leeftijd']]      # DataFrame
type(df[['naam', 'leeftijd']]) # <class 'pandas.core.frame.DataFrame'>

# Dot notation (alleen voor geldige Python identifiers)
df.naam                       # Series (werkt alleen als kolom geen spaties/speciale tekens heeft)
df.leeftijd                   # Series

# Kolom slicing (werkt alleen bij gesorteerde kolommen)
df.loc[:, 'naam':'stad']      # Van kolom 'naam' tot 'stad'

# Alle kolommen behalve één
df.drop('salaris', axis=1)    # Alle kolommen behalve 'salaris'
df.loc[:, df.columns != 'salaris']  # Alternatieve methode
```

#### Rij Selectie - .loc (Label-based)
```python
# Enkele rij (geeft Series terug)
df.loc['emp1']                # Series van eerste employee
type(df.loc['emp1'])          # <class 'pandas.core.series.Series'>

# Meerdere rijen (geeft DataFrame terug)
df.loc[['emp1', 'emp3']]      # DataFrame met emp1 en emp3
df.loc['emp1':'emp3']         # Van emp1 tot emp3 (INCLUSIEF emp3!)

# Rij slicing met step
df.loc['emp1':'emp4':2]       # emp1, emp3 (elke tweede)

# Alle rijen
df.loc[:]                     # Alle rijen (equivalent aan df)
```

#### Rij Selectie - .iloc (Position-based)
```python
# Enkele rij op positie
df.iloc[0]                    # Eerste rij (Series)
df.iloc[-1]                   # Laatste rij (Series)

# Meerdere rijen op posities
df.iloc[[0, 2]]               # Rijen op posities 0 en 2
df.iloc[1:3]                  # Posities 1 en 2 (EXCLUSIEF 3!)

# Rij slicing
df.iloc[:2]                   # Eerste twee rijen
df.iloc[2:]                   # Vanaf derde rij
df.iloc[::2]                  # Elke tweede rij
df.iloc[::-1]                 # Alle rijen omgekeerd
```

#### Cel Selectie - Specifieke Waarden
```python
# Enkele cel
df.loc['emp1', 'naam']        # 'Alice'
df.iloc[0, 0]                 # 'Alice' (positie 0,0)

# .at en .iat voor snellere toegang tot enkele cel
df.at['emp1', 'naam']         # Sneller dan .loc voor enkele waarde
df.iat[0, 0]                  # Sneller dan .iloc voor enkele waarde

# Meerdere cellen
df.loc['emp1', ['naam', 'leeftijd']]  # Series van specifieke kolommen
df.iloc[0, [0, 1]]            # Zelfde met posities
```

#### 2D Selectie - Rijen EN Kolommen
```python
# Rechthoekige selectie
df.loc['emp1':'emp2', 'naam':'stad']    # Rijen emp1-emp2, kolommen naam-stad
df.iloc[0:2, 0:3]                       # Eerste 2 rijen, eerste 3 kolommen

# Gemengde selectie
df.loc[['emp1', 'emp3'], 'naam':'stad'] # Specifieke rijen, kolom range
df.iloc[[0, 2], 1:]                     # Specifieke rijen, vanaf kolom 1

# Hele rijen of kolommen
df.loc[:, 'naam']             # Alle rijen, kolom 'naam'
df.loc['emp1', :]             # Rij 'emp1', alle kolommen
df.iloc[:, 0]                 # Alle rijen, eerste kolom
df.iloc[0, :]                 # Eerste rij, alle kolommen
```

#### Boolean Indexing - Conditionele Selectie
```python
# Basis boolean indexing
mask = df['leeftijd'] > 30
df[mask]                      # Rijen waar leeftijd > 30
df.loc[mask]                  # Equivalent, maar explicieter

# Multiple conditions (BELANGRIJK: gebruik & en | niet 'and'/'or')
young_high_salary = (df['leeftijd'] < 30) & (df['salaris'] > 51000)
df[young_high_salary]

# Complex conditions
df[(df['leeftijd'] > 25) & 
   (df['stad'].isin(['Brussel', 'Gent'])) & 
   (df['salaris'] >= 50000)]

# Boolean indexing met .loc voor specifieke kolommen
df.loc[df['leeftijd'] > 30, ['naam', 'stad']]
df.loc[df['stad'] == 'Brussel', 'salaris'] = 52000  # Assignment
```

#### Query Method - SQL-like Selection
```python
# Query methode voor readable conditions
df.query('leeftijd > 30')
df.query('leeftijd > 30 and stad == "Gent"')
df.query('salaris >= 50000 and leeftijd < 35')

# Met variabelen
min_leeftijd = 30
df.query('leeftijd > @min_leeftijd')

# String operations in query
df.query('stad.str.contains("ent")')
df.query('naam.str.len() > 5')
```

### MultiIndex - Hierarchische Indexing

#### MultiIndex Series
```python
# MultiIndex Series maken
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
multi_series = pd.Series([10, 20, 30, 40], index=arrays)
print(multi_series)
# A  1    10
#    2    20
# B  1    30
#    2    40

# MultiIndex toegang
multi_series['A']             # Alle waarden voor level 0 = 'A'
multi_series['A', 1]          # Specifieke combinatie
multi_series.loc['A']         # Series met level 0 = 'A'
multi_series.loc[('A', 1)]    # Specifieke tuple
```

#### MultiIndex DataFrame
```python
# MultiIndex DataFrame maken
arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo'],
          ['one', 'two', 'one', 'two', 'one', 'two']]
df_multi = pd.DataFrame(np.random.randn(6, 2), 
                       index=arrays, 
                       columns=['A', 'B'])

# MultiIndex toegang
df_multi.loc['bar']           # Alle rijen waar level 0 = 'bar'
df_multi.loc[('bar', 'one')]  # Specifieke combinatie
df_multi.loc['bar':'baz']     # Slice op level 0

# Cross-section
df_multi.xs('one', level=1)   # Alle rijen waar level 1 = 'one'
df_multi.xs(('bar', 'one'))   # Specifieke tuple

# Swaplevel en sort
df_multi.swaplevel(0, 1).sort_index()
```

### Advanced Indexing Technieken

#### Index Manipulation
```python
# Index resetten
df_reset = df.reset_index()           # Index wordt kolom
df_new_index = df.reset_index(drop=True)  # Index wegdoen

# Index instellen
df.set_index('naam')                  # Kolom 'naam' als index
df.set_index(['stad', 'naam'])        # MultiIndex van kolommen

# Index wijzigen
df.index = ['nieuwe', 'index', 'labels', 'hier']
df.rename(index={'emp1': 'employee_1'})
```

#### Index Operations
```python
# Index informatie
df.index.name = 'employee_id'        # Index een naam geven
df.index.names                       # Index namen (voor MultiIndex)
df.index.nlevels                     # Aantal index levels
df.index.is_unique                   # Check of index uniek is

# Index sorting
df.sort_index()                      # Sorteer op index
df.sort_index(ascending=False)       # Omgekeerd sorteren
df.sort_index(axis=1)                # Sorteer kolommen

# Index types
df.index.dtype                       # Type van index
pd.RangeIndex(0, 5)                  # Efficient voor integer sequences
pd.DatetimeIndex(['2023-01-01', '2023-01-02'])  # Voor dates
```

#### Performance Considerations
```python
# Index performance
df_large = pd.DataFrame(np.random.randn(100000, 4), 
                       columns=['A', 'B', 'C', 'D'])

# Sorted index is sneller voor slicing
df_sorted = df_large.sort_index()
%timeit df_sorted.loc[1000:2000]     # Sneller
%timeit df_large.loc[1000:2000]      # Langzamer

# Unique index is sneller voor lookups
df_unique = df_large.reset_index().set_index('index')
```

### Indexing Best Practices

#### Do's and Don'ts
```python
# ✅ GOED: Gebruik .loc en .iloc expliciet
df.loc[df['leeftijd'] > 30, 'naam']
df.iloc[0:5, 1:3]

# ❌ SLECHT: Chained indexing
# df['leeftijd'][df['naam'] == 'Alice'] = 26  # SettingWithCopyWarning!

# ✅ GOED: Gebruik .loc voor assignments
df.loc[df['naam'] == 'Alice', 'leeftijd'] = 26

# ✅ GOED: Boolean operations met parentheses
df[(df['leeftijd'] > 25) & (df['salaris'] < 55000)]

# ❌ SLECHT: Zonder parentheses
# df[df['leeftijd'] > 25 & df['salaris'] < 55000]  # Error!

# ✅ GOED: .copy() bij subset assignments
df_subset = df[df['leeftijd'] > 30].copy()
df_subset.loc[:, 'nieuwe_kolom'] = 'waarde'

# ✅ GOED: isin() voor membership
df[df['stad'].isin(['Brussel', 'Gent'])]

# ❌ SLECHT: Multiple equality checks
# df[(df['stad'] == 'Brussel') | (df['stad'] == 'Gent')]  # Werkt, maar verbose
```

#### Performance Tips
```python
# Voor grote datasets
# 1. Gebruik categoricals voor repetitive strings
df['stad'] = df['stad'].astype('category')

# 2. Set index voor frequent lookups
df_indexed = df.set_index('naam')
df_indexed.loc['Alice']  # Sneller dan df[df['naam'] == 'Alice']

# 3. Gebruik .at/.iat voor single value access
value = df.at['emp1', 'naam']  # Sneller dan df.loc['emp1', 'naam']

# 4. Boolean indexing vs query
# Voor simple conditions: boolean indexing sneller
df[df['leeftijd'] > 30]
# Voor complex conditions: query readable maar langzamer
df.query('leeftijd > 30 and salaris > 50000')
```

## Data Manipulatie

### Kolommen Toevoegen/Verwijderen
```python
# Kolom toevoegen
df['new_col'] = df['A'] + df['B']
df['constant'] = 5
df.insert(1, 'inserted_col', [10, 20, 30])  # Op specifieke positie

# Kolom verwijderen
df.drop('column_name', axis=1, inplace=True)  # inplace=True wijzigt origineel
df_new = df.drop(['col1', 'col2'], axis=1)    # Meerdere kolommen
del df['column_name']  # Alternatieve methode
```

### Rijen Toevoegen/Verwijderen
```python
# Rij toevoegen
new_row = pd.Series({'A': 4, 'B': 5, 'C': 6}, name='new_row')
df = pd.concat([df, new_row.to_frame().T])

# Rijen verwijderen
df.drop('row_name', axis=0, inplace=True)
df.drop(['row1', 'row2'], axis=0, inplace=True)  # Meerdere rijen
df = df[df['A'] > 1]  # Conditioneel verwijderen
```

### Data Transformatie
```python
# Waarden vervangen
df.replace('old_value', 'new_value')
df.replace({'col1': {'old': 'new'}})  # Specifieke kolom
df.replace([1, 2], [10, 20])          # Meerdere waarden

# Waarden mappen
mapping = {'A': 1, 'B': 2, 'C': 3}
df['new_col'] = df['old_col'].map(mapping)

# Apply functies
df['new_col'] = df['old_col'].apply(lambda x: x * 2)
df['new_col'] = df['old_col'].apply(my_function)

# Waarden sorteren
df.sort_values('column')                    # Oplopend
df.sort_values('column', ascending=False)   # Aflopend
df.sort_values(['col1', 'col2'])           # Meerdere kolommen
df.sort_index()                            # Op index sorteren
```

## Missing Values

### Detecteren van Missing Values
```python
# Missing values checken
df.isnull()         # Boolean DataFrame
df.notnull()        # Inverse van isnull()
df.isnull().any()   # Kolommen met missing values
df.isnull().sum()   # Aantal missing values per kolom

# Specifieke kolom
df['column'].isnull().sum()

# Rijen met missing values
df[df.isnull().any(axis=1)]
```

### Omgaan met Missing Values
```python
# Rijen/kolommen met NaN verwijderen
df.dropna()                    # Alle rijen met NaN
df.dropna(axis=1)              # Alle kolommen met NaN
df.dropna(subset=['col1'])     # Alleen rijen met NaN in col1
df.dropna(thresh=2)            # Behoud rijen met min 2 non-NaN waarden

# Missing values vullen
df.fillna(0)                   # Alle NaN met 0
df.fillna(method='ffill')      # Forward fill
df.fillna(method='bfill')      # Backward fill
df.fillna(df.mean())           # Met gemiddelde
df['col'].fillna(df['col'].mean())  # Specifieke kolom

# Interpolatie
df.interpolate()               # Lineaire interpolatie
df.interpolate(method='polynomial', order=2)  # Polynomiale interpolatie
```

## Operaties en Berekeningen

### Wiskundige Operaties
```python
# Basis operaties
df['A'] + df['B']       # Element-wise optelling
df['A'] * 2             # Scalaire vermenigvuldiging
df.sum()                # Som per kolom
df.sum(axis=1)          # Som per rij
df.mean()               # Gemiddelde per kolom
df.std()                # Standaarddeviatie per kolom

# Statistische functies
df.min()                # Minimum
df.max()                # Maximum
df.median()             # Mediaan
df.quantile(0.25)       # Kwartiel
df.var()                # Variantie
df.corr()               # Correlatiematrix
```

### Operaties tussen DataFrames
```python
# Index alignment bij operaties
df1 + df2               # Automatische index alignment
df1.add(df2, fill_value=0)  # Met fill_value voor missing indices

# Operaties tussen DataFrame en Series
df + series             # Series wordt uitgezonden over kolommen
df.add(series, axis=0)  # Expliciet specificeren van as
```

## GroupBy Operaties

### Basic GroupBy
```python
# Groeperen
grouped = df.groupby('category')
grouped = df.groupby(['cat1', 'cat2'])  # Meerdere kolommen

# Basis aggregaties
df.groupby('category').sum()
df.groupby('category').mean()
df.groupby('category').count()
df.groupby('category').size()           # Aantal rijen per groep
df.groupby('category')['value'].sum()   # Specifieke kolom
```

### Geavanceerde GroupBy
```python
# Meerdere aggregaties tegelijk
df.groupby('category').agg(['min', 'max', 'mean'])

# Verschillende aggregaties per kolom
df.groupby('category').agg({
    'value1': 'sum',
    'value2': ['mean', 'std'],
    'value3': 'count'
})

# Custom aggregatie functies
def custom_agg(x):
    return x.max() - x.min()

df.groupby('category').agg(custom_agg)

# Apply met custom functies
def normalize(group):
    return (group - group.mean()) / group.std()

df.groupby('category').apply(normalize)

# Transform (behoudt originele shape)
df.groupby('category').transform('mean')

# Filter groepen
df.groupby('category').filter(lambda x: len(x) > 5)
df.groupby('category').filter(lambda x: x['value'].sum() > 100)
```

### Itereren over Groepen
```python
# Over groepen itereren
for name, group in df.groupby('category'):
    print(f"Group {name}:")
    print(group)
    print()

# Specifieke groep ophalen
df.groupby('category').get_group('group_name')
```

## Merge en Join

### Merge
```python
# Basic merge (inner join op gemeenschappelijke kolommen)
pd.merge(df1, df2)

# Specificeren van join kolom
pd.merge(df1, df2, on='key')
pd.merge(df1, df2, on=['key1', 'key2'])

# Verschillende kolomnamen
pd.merge(df1, df2, left_on='key1', right_on='key2')

# Merge types
pd.merge(df1, df2, how='inner')    # Standaard
pd.merge(df1, df2, how='outer')    # Alle rijen
pd.merge(df1, df2, how='left')     # Alle rijen van df1
pd.merge(df1, df2, how='right')    # Alle rijen van df2

# Merge op index
pd.merge(df1, df2, left_index=True, right_index=True)
pd.merge(df1, df2, left_index=True, right_on='key')

# Overlappende kolomnamen
pd.merge(df1, df2, on='key', suffixes=['_left', '_right'])
```

### Join
```python
# Join (standaard left join op index)
df1.join(df2)
df1.join(df2, how='outer')
df1.join(df2, rsuffix='_right')

# Join met merge equivalent
df1.join(df2)  # = pd.merge(df1, df2, left_index=True, right_index=True, how='left')
```

### Concatenate
```python
# Rijen samenvoegen
pd.concat([df1, df2])                    # Onder elkaar
pd.concat([df1, df2], ignore_index=True) # Nieuwe index

# Kolommen samenvoegen
pd.concat([df1, df2], axis=1)            # Naast elkaar
```

## String Operaties

### Basis String Methoden
```python
# Vectorized string operaties met .str accessor
df['text'].str.lower()           # Naar lowercase
df['text'].str.upper()           # Naar uppercase
df['text'].str.title()           # Title case
df['text'].str.capitalize()      # Eerste letter hoofdletter

# String eigenschappen
df['text'].str.len()             # Lengte van strings
df['text'].str.count('a')        # Tel karakter 'a'
df['text'].str.startswith('A')   # Begint met 'A'
df['text'].str.endswith('ing')   # Eindigt met 'ing'
```

### String Manipulatie
```python
# Slicing en indexing
df['text'].str[0]                # Eerste karakter
df['text'].str[:5]               # Eerste 5 karakters
df['text'].str[-1]               # Laatste karakter

# Trimmen en padding
df['text'].str.strip()           # Whitespace wegsnijden
df['text'].str.lstrip()          # Links trimmen
df['text'].str.rstrip()          # Rechts trimmen
df['text'].str.pad(10, side='left', fillchar='0')  # Padding

# Vervangen
df['text'].str.replace('old', 'new')
df['text'].str.replace('\\d+', 'NUMBER', regex=True)  # Regex replace
```

### String Splitten en Samenvoegen
```python
# Splitten
df['text'].str.split()                    # Split op whitespace
df['text'].str.split(',')                 # Split op komma
df['text'].str.split(',', expand=True)    # Split naar kolommen
df['text'].str.split(',', n=1)            # Max 1 split

# Samenvoegen
df['text1'].str.cat(df['text2'])          # Concateneren
df['text1'].str.cat(df['text2'], sep=' ') # Met separator
df['text'].str.cat(sep=' ')               # Alle waarden samenvoegen
```

### Regex Operaties
```python
# Patroon matching
df['text'].str.contains('pattern')              # Bevat patroon
df['text'].str.contains('^A', regex=True)       # Begint met A
df['text'].str.match(r'\\d+')                   # Match vanaf begin
df['text'].str.findall(r'\\d+')                 # Vind alle matches

# Extractie
df['text'].str.extract(r'(\\d+)')               # Eerste groep
df['text'].str.extractall(r'(\\d+)')            # Alle groepen
```

## Pivot Tables en Reshaping

### Pivot Tables
```python
# Basic pivot
df.pivot_table(values='value', index='row', columns='col')
df.pivot_table(values='value', index='row', columns='col', aggfunc='mean')

# Meerdere aggregaties
df.pivot_table(values='value', index='row', columns='col', 
               aggfunc=['sum', 'mean', 'count'])

# Pivot met meerdere kolommen
df.pivot_table(values=['val1', 'val2'], index='row', columns='col')
```

### Melt (Long Format)
```python
# Wide naar long format
df.melt(id_vars=['id'], value_vars=['col1', 'col2'])
df.melt(id_vars=['id'], var_name='variable', value_name='value')
pd.melt(df, id_vars=['id'])
```

### Stack en Unstack
```python
# Stack: kolommen naar rijen
df.stack()
df.stack(level=0)

# Unstack: rijen naar kolommen
df.unstack()
df.unstack(level=0)
```

## Time Series

### Datetime Handling
```python
# Datetime conversie
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Datetime properties
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
```

### Time Series Indexing
```python
# Datetime index
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# Time-based selection
df['2023']                    # Heel jaar 2023
df['2023-01']                 # Januari 2023
df['2023-01-01':'2023-01-31'] # Datum range
```

### Resampling
```python
# Resample naar verschillende frequenties
df.resample('D').mean()       # Dagelijks gemiddelde
df.resample('W').sum()        # Wekelijkse som
df.resample('M').last()       # Laatste waarde per maand
df.resample('Q').agg({'col1': 'sum', 'col2': 'mean'})  # Kwartaal aggregatie
```

## Performance Tips

### Memory Optimization
```python
# Datatypes optimaliseren
df['int_col'] = df['int_col'].astype('int32')    # Kleinere integers
df['cat_col'] = df['cat_col'].astype('category') # Categoricals voor strings
df['bool_col'] = df['bool_col'].astype('bool')   # Booleans

# Memory usage checken
df.info(memory_usage='deep')
df.memory_usage(deep=True)
```

### Efficient Operations
```python
# Vectorized operaties i.p.v. loops
# SLECHT:
for i in range(len(df)):
    df.loc[i, 'new_col'] = df.loc[i, 'col1'] * 2

# GOED:
df['new_col'] = df['col1'] * 2

# Method chaining
result = (df
          .query('col1 > 5')
          .groupby('category')
          .agg({'value': 'sum'})
          .reset_index())

# Gebruik .loc voor assignments
df.loc[df['A'] > 5, 'B'] = 'high'  # Geen SettingWithCopyWarning
```

## Veelgemaakte Fouten

### Index Verwarring
```python
# FOUT: Gebruik van .ix (deprecated)
# df.ix[0, 'col']  # NIET GEBRUIKEN

# GOED: Gebruik .loc of .iloc
df.loc[0, 'col']    # Label-based
df.iloc[0, 0]       # Position-based
```

### Chained Indexing
```python
# FOUT: Chained indexing
# df['col'][0] = 'new_value'  # Kan SettingWithCopyWarning geven

# GOED: Gebruik .loc
df.loc[0, 'col'] = 'new_value'
```

### Copy vs View
```python
# Expliciete copy maken
df_copy = df.copy()

# Controleren of het een view of copy is
df_subset = df[df['A'] > 5]
df_subset.loc[0, 'B'] = 'new'  # Gebruik .loc voor veiligheid
```

### Performance Valkuilen
```python
# SLECHT: Iteratieve append
df_new = pd.DataFrame()
for i in range(1000):
    df_new = df_new.append({'col': i}, ignore_index=True)  # Zeer langzaam!

# GOED: Lijst opbouwen dan DataFrame maken
data_list = []
for i in range(1000):
    data_list.append({'col': i})
df_new = pd.DataFrame(data_list)

# BETER: Gebruik pd.concat voor meerdere DataFrames
df_list = [df1, df2, df3]
df_combined = pd.concat(df_list, ignore_index=True)
```

## Debugging en Troubleshooting

### Data Inspection
```python
# Snel overzicht
df.head()
df.tail()
df.sample(5)                    # Random sample
df.describe(include='all')

# Specifieke issues
df.duplicated().sum()           # Aantal duplicaten
df.drop_duplicates()            # Duplicaten verwijderen
df.isna().sum()                 # Missing values per kolom
df.dtypes                       # Data types checken
```

### Performance Monitoring
```python
# Geheugengebruik
df.memory_usage(deep=True)
df.info(memory_usage='deep')

# Timing operations
import time
start = time.time()
# Your operation here
end = time.time()
print(f"Operation took {end - start:.2f} seconds")

# Profiling met %timeit in Jupyter
# %timeit df.groupby('col').sum()
```