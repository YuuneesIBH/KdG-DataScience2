Belangrijk, we werken hier met 1dimensionale arrays dus geen dataframes maar Series
## Data voorbereiden
```python
import pandas as pd
data = pd.read_csv('../../datasets/file.csv', sep=",", decimal=".", header=None)

x = data
y = data['target']
```

### Shapes 
```python
# X toont ons de samples en y toont de features
print(x.shape, y.shape)
```

### Kijken of ze NaN values hebben
```python
# pandas
print(pd.isna(x).any())
print(pd.isna(y).any())

# numpy
print(np.isnan(x).any())  
print(np.isnan(y).any())
```

### Kijken hoeveel verschillende classes er zijn
```python
# pandas
pd.unique(y)

# numpy
np.unique(y)
```

## Data normaliseren
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(x)
```

## Data opsplitsen in training en test data
```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_norm, wine_y, train_size=0.85)
```

## Data omzetten naar one-hot-encoding
```python
from tensorflow.keras.utils import to_categorical

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
print(y_train_one_hot.shape, y_test_one_hot.shape)
```

## Model voorbereiden

### Categorical
Hier zullen we de input layer moeten voorbereiden met het correct aantal nodes -> **NEEM AANTAL FEATURES** - kijk hiervoor terug naar de x.shape en neem het aantal kolommen
```python
from tensorflow.keras import Model  
from tensorflow.keras.layers import Input, Dense  
from tensorflow.keras.optimizers import Adam

# Deze shape is het aantal kolommen dat de data heeft
inputs = Input(shape=(13,))

# De verschillende hidden layers hier kun je kiezen hoeveel je doet
# Ook belangrijk is dat we hier de activation methode zullen kiezen
x = Dense(128, activation='relu')(inputs) # Hier geven we eerst de inputs die we zelf hebben gezet mee
x = Dense(64, activation='relu')(x) # Hier geven we de output van de vorige layer mee

# Aantal neuronen in de output MOET overeen komen met het aantal unieke klassen van de target
# Om dit te vinden kun je 
total_output_neurons = pd.unique(y).size
# softmax gebruiken we als we met categorical values werken, linear voor getallen
outputs = Dense(total_output_neurons, activation='softmax')(x) # Bij de laatste kies je opnieuw de gewenste activation function en geef je de vorige output als input

# dit geven we nu allemaal aan het mddel mee
model = Model(inputs, outputs, name="MNIST")
model.summary() # Geeft een samenvatting
# We kiezen Adam als algoritme, 
# Als loss bij categorical gegevens gebruiken we 'categorical_crossentropy' EN nemen we als metric de accuracy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
Nu is het model volledig getraind voor categorical gegevens

### Regression
Als we regressie-modellen willen gebruiken zullen we een aantal dingen anders moeten doen

Let op: het begin is exact hetzelfde
```python
from tensorflow.keras import Model  
from tensorflow.keras.layers import Input, Dense  
from tensorflow.keras.optimizers import Adam

# Deze shape is het aantal kolommen dat de data heeft
inputs = Input(shape=(13,))

# De verschillende hidden layers hier kun je kiezen hoeveel je doet
# Ook belangrijk is dat we hier de activation methode zullen kiezen
x = Dense(128, activation='relu')(inputs) # Hier geven we eerst de inputs die we zelf hebben gezet mee
x = Dense(64, activation='relu')(x) # Hier geven we de output van de vorige layer mee

# Aantal neuronen in de output MOET overeen komen met het aantal unieke klassen van de target
# Om dit te vinden kun je 
total_output_neurons = pd.unique(y).size
# Hier gebruiken we dan als activation linear
outputs = Dense(total_output_neurons, activation='linear')(x)  
model = Model(inputs, outputs, name="HOUSE")  
model.summary()  
# En hier is het belangrijk om te weten dat we als loss mse nemen en als metric gebruiken we de mape
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error',  
              metrics=['mean_absolute_percentage_error'])
```

Hier zien we ze allemaal staan: **regression**, **classification** en **binary classification** (kans is klein dat we deze laatste zullen moeten gebruiken)
![[Pasted image 20250612192840.png]]
## Model trainen
```python
from plot_loss import plot_loss

# Hier geven we aan ons model de training data mee, hoeveel epochs, hoe groot iedere batch size en de stappen die we zullen zetten
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validaiton_split=0.1)

# Zal de loss tonen, zie je spikes in de validation maar niet de training dan is er overfitting. Doen ze exact hetzelfde is het goed
plot_loss(history)
```

## Model evalueren
```python
# Indien met klassen
model.evaluate(x_test, y_test_one_hot)

# Indien met regression
model.evaluate(x_test, y_test)
```
Dit zal ons de loss en de accuracy tonen. Als deze accuracy 1 is dan weet je dat hij aan overfitting heeft gedaan

## Predicting
Als je nu wilt voorspellen met dit model kan dat door nieuwe data te nemen
```python
new_data = [[12, 6.5, 3, 25, 100, 2.5, 4, 0.5, 2, 8, 1, 3, 500]]
new_data_norm = scaler.transform(new_data)
predicted_outcome = model.predict(new_data_norm)
print(predicted_outcome)  # Zal dan de uitkomst voorspellen, kan de accuracy zijn indien categorical of kan ook een MAPE zijn indien regression 
```
- Zal bij regression een value geven die hij voorspelt en bij categorical voor iedere categorie een accuracy met de waarschijnlijkheid. Daar waar deze value het hoogste is is het waarschijnlijkst

EXTRA: (andere evaluatietechnieken)

### ReLU (huidige keuze) - meest gebruikt
hidden = Dense(8, activation='relu')(norm)

### Leaky ReLU - voorkomt "dying ReLU" probleem
hidden = Dense(8, activation='leaky_relu')(norm)

### ELU - Exponential Linear Unit
hidden = Dense(8, activation='elu')(norm)

### Swish/SiLU - moderne activatiefunctie
hidden = Dense(8, activation='swish')(norm)

### Tanh - klassieke functie
hidden = Dense(8, activation='tanh')(norm)

### Sigmoid - kan leiden tot vanishing gradients
hidden = Dense(8, activation='sigmoid')(norm)