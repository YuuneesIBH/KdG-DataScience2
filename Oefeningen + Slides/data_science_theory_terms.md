# Data Science 2 - Theorie en Termen

## Machine Learning

### Definitie
Machine learning is het proces van automatisch patronen vinden in data zonder expliciete programmering voor elke specifieke taak.

### Belangrijkste Verschillen met Traditionele Analyse
- **Traditionele Data Analyse**: Formuleer hypothese → Verzamel data → Verken data → Bouw model → Test significantie → Trek conclusies
- **Machine Learning**: Verzamel data → Train model → Test model → Trek conclusies

### Types Machine Learning Problemen

#### Supervised Learning
Leren met gelabelde data (voorbeelden met bekende uitkomsten)

**Waarde Schatting/Regressie**
- Voorspel continue variabelen
- Voorbeelden: verkoopvoorspelling, auto-gebruiksvoorspelling

**Classificatie**
- Voorspel categorische variabelen
- Voorbeelden: churn voorspelling, medische diagnose

#### Unsupervised Learning
Leren zonder gelabelde data

**Clustering/Segmentatie**
- Groepeer gelijkaardige cases/observaties
- Voorbeelden: klantensegmentatie, document topic search

**Associatieregel Ontdekking**
- Vind gebeurtenissen die samen voorkomen
- Voorbeelden: winkelmandje analyse, aanbevelingssystemen

### Data Voorbereiding

#### Feature Matrix (X)
- 2-dimensionale structuur met onafhankelijke features/predictors
- Rijen vertegenwoordigen voorbeelden, kolommen vertegenwoordigen features
- Moet 2-dimensionaal zijn in scikit-learn, zelfs voor enkele features

#### Target Array (y)
- Vector met labels/afhankelijke variabelen
- Meestal 1-dimensionaal voor enkele target features
- Kan 2-dimensionaal zijn voor meerdere target features

#### Normalisatie
Schaling van features om gelijkaardige grootteordes en distributies te hebben

**StandardScaler (Z-scores)**
- Transformeer data naar gemiddelde = 0, standaarddeviatie = 1
- Formule: `x_norm = (x - gemiddelde) / std_dev`

**MinMaxScaler**
- Transformeer data naar bereik [0,1]
- Formule: `x_norm = (x - min) / (max - min)`

### Model Validatie

#### Training vs Testen
- **Training Set**: Gebruikt om model te bouwen (typisch 70-80% van data)
- **Test Set**: Gebruikt om modelprestaties te evalueren (typisch 20-30% van data)
- **Validatie Set**: Gebruikt voor hyperparameter tuning (optionele derde split)

#### Overfitting vs Underfitting
- **Overfitting**: Model presteert goed op trainingsdata maar slecht op nieuwe data (te complex)
- **Underfitting**: Model presteert slecht op zowel training- als testdata (te simpel)
- **Generalisatie**: Vermogen van model om goed te presteren op nieuwe, ongeziene data

#### Cross-Validatie
- Splits data in meerdere partities (folds)
- Train op N-1 partities, test op overgebleven partitie
- Herhaal N keer met verschillende testpartities
- Bereken gemiddelde prestatie over alle folds

#### Validatiemetrieken

**Regressie Metrieken**
- **MAE**: Mean Absolute Error (Gemiddelde Absolute Fout)
- **MAPE**: Mean Absolute Percentage Error (Gemiddelde Absolute Percentage Fout)
- **RMSE**: Root Mean Squared Error (Wortel Gemiddelde Gekwadrateerde Fout)
- **R²**: R-kwadraat (determinatiecoëfficiënt)

**Classificatie Metrieken**
- **Accuracy**: Percentage correcte voorspellingen
- **Precision**: Ware positieven / (Ware positieven + Valse positieven)
- **Recall**: Ware positieven / (Ware positieven + Valse negatieven)
- **F1-Score**: Harmonisch gemiddelde van precision en recall
- **Confusion Matrix**: Tabel die correcte en incorrecte voorspellingen toont

### Hyperparameters
Parameters die het leeralgoritme controleren maar niet geleerd worden uit data. Voorbeelden:
- Boomdiepte in decision trees
- Aantal clusters in K-means
- Leersnelheid in neural networks
- Graad in polynomiale regressie

### Model Selectie
Proces van het kiezen van het beste algoritme en hyperparameters door systematische vergelijking met validatietechnieken.

## Specifieke Algoritmen

### Lineaire Regressie
- Past lineaire relatie: `y = b + a₁x₁ + a₂x₂ + ... + aₙxₙ`
- **Coëfficiënten**: Hellingen (a₁, a₂, ...) en intercept (b)
- Gebruikt voor voorspelling van continue waarden

### Polynomiale Regressie
- Past polynomiale relatie: `y = a₀ + a₁x + a₂x² + ... + aₙxⁿ`
- **Graad**: Orde van polynoom (n)
- Kan niet-lineaire relaties vastleggen

### Decision Trees
- Boom-achtig model dat beslissingen neemt gebaseerd op feature waarden
- **Nodes**: Beslissingspunten met splitscondities
- **Leaves**: Finale voorspellingen
- **Criteria**: Gini impurity of entropie voor meten van splitkwaliteit
- **Pruning Parameters**: max_depth, min_samples_split, min_samples_leaf

### Random Forests
- Ensemble methode die meerdere decision trees combineert
- **n_estimators**: Aantal bomen in het bos
- Vermindert overfitting vergeleken met enkele decision trees

### K-Means Clustering
- Verdeelt data in k clusters gebaseerd op feature gelijkenis
- **Centroids**: Middelpunten van clusters
- **k**: Aantal clusters (moet gespecificeerd worden)
- **Initialisatie**: k-means++ of willekeurige centroid plaatsing

## Neurale Netwerken

### Structuur
- **Input Laag**: Ontvangt input features
- **Verborgen Lagen**: Verwerkt informatie (Deep Learning wanneer >1 verborgen laag)
- **Output Laag**: Produceert finale voorspellingen
- **Gewichten**: Verbindingsterktes tussen neuronen
- **Biases**: Drempelwaarden voor neuron activatie

### Neuron Componenten
- **Integratie Functie**: Gewogen som van inputs plus bias
- **Activatie Functie**: Transformeert geïntegreerd signaal
  - Linear, Sigmoid, ReLU, Tanh, Softmax, etc.

### Training Proces
- **Forward Pass**: Input stroomt door netwerk om output te produceren
- **Loss Functie**: Meet verschil tussen voorspelde en werkelijke waarden
  - MSE, MAE, Binary/Categorical Cross-entropy
- **Backpropagation**: Past gewichten aan gebaseerd op fouten
- **Optimizer**: Algoritme voor het updaten van gewichten (Adam, SGD)
- **Epochs**: Volledige passages door trainingsdata
- **Batches**: Subsets van data die samen verwerkt worden

### Belangrijke Concepten
- **Leersnelheid**: Stapgrootte voor gewichtupdates
- **Overfitting**: Model memoriseert trainingsdata maar faalt op nieuwe data
- **One-Hot Encoding**: Converteren van categorieën naar binaire vectors
- **Softmax**: Activatie die ervoor zorgt dat output kansen optellen tot 1

## Meta-heuristieken

### Definitie
High-level procedures toepasbaar op elk optimalisatieprobleem, ontworpen om goede (niet noodzakelijk optimale) oplossingen te vinden.

### Optimalisatieprobleem Componenten
- **Variabelen**: Beslissingsparameters die bepaald moeten worden
- **Oplossingsruimte**: Verzameling van alle mogelijke oplossingen
- **Beperkingen**: Limitaties op geldige oplossingen
- **Doelfunctie**: Functie om te maximaliseren of minimaliseren

### Lokale Zoektocht vs Globale Zoektocht
- **Lokale Zoektocht**: Verkent oplossingen in de "buurt" van huidige oplossing
- **Lokaal Optimum**: Beste oplossing in lokaal gebied (mogelijk niet globaal beste)
- **Globaal Optimum**: Beste oplossing over hele oplossingsruimte

### Simulated Annealing
Meta-heuristiek gebaseerd op metallurgie afkoelingsproces
- **Temperatuur**: Controleert acceptatie van slechtere oplossingen
- **Afkoelingsschema**: Hoe temperatuur afneemt over tijd
- **Acceptatiekans**: `exp(-Δ/T)` voor slechtere oplossingen
- **Kristallisatie**: Geleidelijke convergentie naar goede oplossing

### Belangrijke Principes
- Start met initiële oplossing
- Accepteer betere buuroplossingen
- Soms accepteer slechtere oplossingen (ontsnap aan lokale optima)
- Verminder geleidelijk acceptatie van slechtere oplossingen
- Onthoud beste gevonden oplossing

### Parameters
- **Begintemperatuur**: Starttemperatuurwaarde
- **Eindtemperatuur**: Eindtemperatuurwaarde
- **Afkoelingssnelheid**: Snelheid van temperatuurvermindering
- **Iteraties**: Aantal stappen bij elke temperatuur