# Pandas - Uitlegsheet

## Wat is Pandas?

Pandas is een Python library specifiek ontwikkeld voor Data Scientists. Het biedt flexibele datastructuren die gebouwd zijn bovenop NumPy arrays, maar met veel meer functionaliteit voor data-analyse.

## Kernconcepten

### Series vs NumPy Array
```python
# NumPy array: impliciete integer index
np_array = np.array([1, 2, 3, 4])
# Index: 0, 1, 2, 3 (automatisch)

# Pandas Series: expliciete index
pd_series = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# Index: 'a', 'b', 'c', 'd' (zelf gedefinieerd)
```

**Voordeel:** Je kunt betekenisvolle labels gebruiken als index!

### DataFrame = 2D Tabel
Een DataFrame is zoals een Excel-spreadsheet:
- **Rijen** = observaties/records
- **Kolommen** = variabelen/features  
- **Index** = labels voor rijen
- **Columns** = labels voor kolommen

## Waarom Pandas gebruiken?

### 1. **Flexibele Data Import/Export**
- CSV, Excel, JSON, SQL databases, etc.
- Automatische detectie van datatypes
- Omgaan met verschillende separators, headers, etc.

### 2. **Intelligente Missing Data Handling**
- Automatische detectie van missing values (NaN, None)
- Flexibele opties voor omgaan met missing data
- Consistent gedrag over alle operaties

### 3. **Krachtige Indexing**
- Label-based indexing (loc)
- Position-based indexing (iloc)  
- Boolean indexing voor filtering
- Hierarchische indexing mogelijk

### 4. **Geavanceerde Data Manipulatie**
- GroupBy operaties voor aggregatie
- Merge/Join operaties zoals in SQL
- Pivot tables
- Time series functionaliteit

## Indexing - Het Verschil Begrijpen

### `.loc` vs `.iloc`

```python
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])

# .loc gebruikt de EXPLICIETE index (labels)
data.loc[1]     # → 'a' (label 1)
data.loc[1:3]   # → 1:'a', 3:'b' (INCLUSIEF eindpunt!)

# .iloc gebruikt de IMPLICIETE index (posities)  
data.iloc[1]    # → 'b' (positie 1)
data.iloc[1:3]  # → positie 1:'b', 2:'c' (EXCLUSIEF eindpunt!)
```

**Onthoud:** 
- `.loc` = **L**abels (inclusief eindpunt)
- `.iloc` = **I**nteger positions (exclusief eindpunt)

## Missing Values - Praktische Benadering

### Detecteren
```python
df.isnull()     # True waar missing values zijn
df.notnull()    # True waar geldige waarden zijn
```

### Strategieën
1. **Droppen:** `df.dropna()` - Verwijder rijen/kolommen met missing values
2. **Vullen:** `df.fillna(value)` - Vervang missing values met een waarde
3. **Interpoleren:** Voor tijdreeksen, schat waarden tussen bekende punten

**Keuze hangt af van:**
- Hoeveel missing data je hebt
- Waarom de data missing is
- Wat je met de data wilt doen

## GroupBy - Split-Apply-Combine

GroupBy is één van de krachtigste features van Pandas:

```python
# 1. SPLIT: Verdeel data in groepen
grouped = df.groupby('category')

# 2. APPLY: Pas functie toe op elke groep  
result = grouped.sum()

# 3. COMBINE: Combineer resultaten
```

### Praktische toepassingen:
- **Aggregatie:** Gemiddelde verkoop per regio
- **Transformatie:** Normaliseer waarden binnen groepen
- **Filtering:** Houd alleen groepen die aan criteria voldoen

## Merge vs Join - Wanneer Wat?

### Merge
- Meer flexibel
- Kan op verschillende kolommen/indexen
- Specificeer join type (inner, outer, left, right)

### Join  
- Sneller voor index-gebaseerde joins
- Standaard left join
- Minder flexibel maar eenvoudiger syntax

**Regel:** Gebruik `merge()` voor complexe joins, `join()` voor eenvoudige index-joins.

## String Operaties - Vectorized Magic

```python
# In plaats van:
for i in range(len(df)):
    df.loc[i, 'col'] = df.loc[i, 'col'].lower()

# Doe dit:
df['col'] = df['col'].str.lower()
```

De `.str` accessor geeft je toegang tot alle string methoden, maar dan vectorized (toegepast op hele kolom tegelijk).

## Categorische Data - Efficiëntie

```python
# String kolom: veel geheugen
df['color'] = ['red', 'blue', 'red', 'blue', ...]

# Categorische kolom: weinig geheugen  
df['color'] = df['color'].astype('category')
```

**Voordelen:**
- **Minder geheugen** gebruik
- **Snellere** operaties
- **Betere** sortering (logische volgorde)

## Praktische Tips

### 1. **Exploratie workflow**
```python
df.info()        # Overzicht datatypes en missing values
df.describe()    # Statistische samenvatting  
df.head()        # Eerste paar rijen bekijken
df.shape         # Dimensies checken
```

### 2. **Chaining operaties**
```python
# In plaats van meerdere regels:
df_filtered = df[df['col'] > 5]
df_grouped = df_filtered.groupby('category')
result = df_grouped.sum()

# Chain ze:
result = (df[df['col'] > 5]
          .groupby('category')
          .sum())
```

### 3. **Performance tips**
- Gebruik `dtype='category'` voor herhalende strings
- Vermijd loops, gebruik vectorized operaties
- `.loc` en `.iloc` zijn sneller dan `.ix`
- GroupBy is vaak sneller dan meerdere filters

## Veelgemaakte Fouten

1. **SettingWithCopyWarning:** Gebruik `.loc` voor assignments
2. **Index confusion:** Let op verschil tussen `.loc` en `.iloc`
3. **Chained indexing:** `df['col'][0]` → gebruik `df.loc[0, 'col']`
4. **Missing values:** Check altijd op NaN values voor berekeningen

## Wanneer Pandas vs NumPy?

**Gebruik Pandas voor:**
- Datasets met labels/names
- Missing data handling
- Data van verschillende types
- Complexe data manipulatie

**Gebruik NumPy voor:**
- Numerieke berekeningen
- Homogene data (alles hetzelfde type)
- Performance-kritische operaties
- Wiskundige operaties