# Data Visualization - Uitlegsheet

## Waarom Data Visualisatie?

Data visualisatie transformeert ruwe data naar visuele context, waardoor:
- **Trends en patronen** makkelijker herkenbaar worden
- **Complexe datasets** vereenvoudigd worden
- Data **toegankelijk en begrijpelijk** wordt voor iedereen
- **Outliers** sneller gespot worden

## Matplotlib vs Seaborn

### Matplotlib
- **Meer controle** over elk detail van je plot
- **Gedetailleerde customisatie** mogelijk
- Basis plotting library in Python
- Vereist meer code voor complexe visualisaties

### Seaborn  
- **Betere standaard esthetiek** - ziet er meteen mooi uit
- **Makkelijker te gebruiken** voor statistische plots
- Gebouwd bovenop Matplotlib
- **Vereenvoudigde syntax** voor complexe visualisaties

## Soorten Grafieken - Wanneer Gebruiken?

### 📈 Line Chart
**Wanneer:** 
- Trends over tijd tonen (tijdreeksen)
- Continue monitoring van data (temperatuur, verkoopcijfers)
- Relatie tussen twee variabelen over tijd

**Juist gebruik:**
- ✅ Verkoop per maand over een jaar
- ✅ Temperatuurveranderingen over dagen
- ❌ Categorieën vergelijken (gebruik bar chart)

### 📊 Scatter Plot  
**Wanneer:**
- **Relaties** tussen twee variabelen onderzoeken
- **Correlaties** visualiseren
- Patronen zoals lineaire/non-lineaire verbanden zoeken

**Voorbeeld:**
- Lengte vs gewicht van personen
- Ervaring vs salaris
- Temperatuur vs ijsverkoop

### 📊 Histogram
**Wanneer:**
- **Verdeling** van continue variabelen tonen
- Geschikt voor interval- of ratio-data
- Frequentie van waarden in bepaalde intervallen

**Verschil met bar chart:**
- Histogram: balken zijn **aan elkaar verbonden** (continue data)
- Bar chart: balken staan **los** (categorische data)

### 📊 Bar Chart
**Wanneer:**
- **Discrete categorieën** vergelijken
- Nominale en ordinale data
- Exacte verschillen tussen categorieën belangrijk zijn

**Types:**
- **Normale bar chart:** Eenvoudige vergelijking
- **Stacked bar chart:** Subcategorieën binnen hoofdcategorieën

### 📦 Box Plot
**Wanneer:**
- **Spreiding** van data visualiseren
- **Outliers** identificeren  
- **Verschillende datasets** vergelijken
- Mediane, kwartielen en extremen tonen

**Toont:**
- Min, Q1, Mediaan, Q3, Max
- Outliers als aparte punten

### 🥧 Pie Chart
**Wanneer:**
- **Delen van een geheel** tonen (100%)
- **Maximaal 5 categorieën** (anders wordt het onduidelijk)
- Relatieve proporties belangrijker dan exacte waarden

**Juist gebruik:**
- ✅ Marktaandeel van 3 bedrijven
- ❌ 10+ categorieën (gebruik bar chart)
- ❌ Exacte verschillen belangrijk (gebruik bar chart)

## Veelgemaakte Fouten

### ❌ Te veel categorieën
- Pie charts met 10+ segmenten worden onleesbaar
- Line charts met te veel lijnen worden chaotisch

### ❌ Y-as manipulatie
- Y-as niet bij 0 laten beginnen kan misleidend zijn
- Overdreven verschillen door schaalmanipulatie

### ❌ Verkeerde grafiek voor datatype
- Pie chart voor data die geen geheel vormt
- Histogram voor categorische data
- Line chart voor niet-tijdgerelateerde categorieën

## Tips voor Betere Grafieken

### ✅ Simpliciteit
- Toon alleen **noodzakelijke informatie**
- Beter meerdere eenvoudige grafieken dan één complexe
- Vermijd "chart junk" - overbodige decoratie

### ✅ Juiste keuzes maken
- **Bar vs Pie:** Bar voor exacte verschillen, Pie voor proporties (max 5 categorieën)
- **Line vs Histogram:** Line voor trends over tijd, Histogram voor verdelingen
- **Box vs Scatter:** Box voor spreiding, Scatter voor relaties tussen variabelen

### ✅ Duidelijke labels
- Altijd titel, x-label en y-label toevoegen
- Legend wanneer nodig
- Eenheden vermelden

## Praktische Richtlijnen

1. **Start met je vraag:** Wat wil je laten zien?
2. **Kies het juiste charttype** op basis van je data en doel
3. **Houd het simpel** - complexiteit vermindert begrip  
4. **Test je grafiek** - begrijpt iemand anders wat je wilt tonen?
5. **Itereer** - eerste versie is zelden de beste

## Snel Beslisschema

```
Wil je trends over tijd tonen? → Line Chart
Wil je relaties tussen variabelen? → Scatter Plot  
Wil je verdeling van data? → Histogram
Wil je categorieën vergelijken? → Bar Chart
Wil je spreiding/outliers? → Box Plot
Wil je delen van geheel (≤5 cat.)? → Pie Chart
```