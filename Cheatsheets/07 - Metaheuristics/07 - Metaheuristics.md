## Inleiding - Optimalisatieproblemen
Een optimalisatieprobleem omvat: 
- Variabelen
- Oplossingsruimte
	- Omschrijving van de verzameling van alle mogelijke oplossingen (oplossing = toekennen van concrete waarden aan variabelen)
	- Lijst van beperkingen: Constraints
- Een te maximaliseren of te minimaliseren doelfunctie (of kostfunctie)

### Een voorbeeld van optimalisatieprobleem
#### Traveling Salesman Problem **TSP**
Typisch voorbeeld van een optimalisatieprobleem, we gaan de kortste route berekenen voor een zakenman om zijn volledige tour af te leggen. Hiervoor moeten we de afstand tussen alle punten berekenen en kunnen we dan in een *distance matrix* zetten. Aan de hand  van deze data kunnen we dan de kortste route berekenen

Belangrijk hier is dat de afstand van A -> B niet perse even lang is als van B -> A omdat er omwegen of eenrichtingstraten zouden kunnen zijn. 

Optimale route berekenen zouden we kunnen doen door *bijvoorbeeld* alle mogelijkheden uit te rekenen en dan de beste te kiezen. Kunnen ook voor iedere stad zien welke route het kortste is

Om te berekenen hoeveel opties dit zijn neem je alle opties en doe je n * n-1 * n-2 tot n = 1
(Hier zou dat 5 * 4 * 3 * 2 * 1 zijn = 120)
-> Valt hier nog mee maar zal snel stijgen
![[Pasted image 20250412172727.png]]

## Algoritme versus heuristiek
Belangrijk verschil 

**Algoritme**: geeft de **OPTIMALE** oplossing, als je het 2 keer laat runnen zal je 2 keer hetzelfde krijgen
**Heuristiek**: zal een benadring gebruiken en de uitkomst zal hiervan afhankelijk zijn. Dus de uitkomst kan verschillen
## Soorten heuristieken
### 'Custom Made'-heuristieken
- Ontwikkeld voor een specifieke optimalisatie-probleem
- Niet herbruikbaar voor andere optimalisatie-problemen
- Gebruikt/Exploiteert specifieke aspecten en eigenschappen niet noodzakelijk aanwezig in andere optimalisatieproblemen
-> Is gewoon specifiek voor 1 probleem

#### Voorbeeld
Stel we nemen terug het Traveling Salesman Probleem
-> Hier zouden we dan één stad/punt kunnen kiezen en dan vanuit dit punt telkens de beste optie kiezen. Stel dat we A kiezen dan zullen we één uitkomst hebben en als we B zouden kiezen kan het zijn dat we een andere uitkomst hebben

*Als er veel steden zijn* zou je dit ook een aantal keer kunnen doen vanuit verschillende startpunten en dan de beste kiezen

### 'Lokale zoek'-Heuristieken
Zal steeds in de 'buurt' zoeken van de vorige oplossing naar een beter oplossing. Een of meerdere stopcriteria worden gehanteerd.
- Slechts één of enkele oplossingen buihouden en die "verbeteren"
- Oplossingen in de buurt zijn oplossingen met "kleine" aanpassingen
- Resultaat is de 'beste' oplossing die tijdens de zoektocht gevonden is
- Risico 'vast' te raken in lokaal minimum
#### Voorbeeld
Stel volgende foto

Stel we beginnen waar "huidige oplossing staat" en we zoeken naar de hoogste piek dan zal die naar rechts gaan en niet naar links want daar is een daling hoewel links wel de hoogste piek is. 

Hoe voorkomen we dat we vast komen zitten?
- Meerdere keren na elkaar uitvoeren met andere initiële oplossingen
- Meta-heuristieken bekijken
![[Pasted image 20250412173830.png]]

#### Ander voorbeeld
Stel volgende foto

Stel de pieken zijn straling en we willen zo weinig mogelijk straling dus we gaan zo diep mogelijk. Dan beginnen we op een plek, kijken we waar er minder straling is en dan zullen we altijd die kant op wandelen, als de straling weer stijgt gaan we terug naar waar die het minste was en zullen we daar weer verder kijken. -> Is zijn voorbeeld van die leden bij de scouts en hun stapjes en vlaggetjes 
![[Pasted image 20250412174115.png]]

### Meta-heuristieken
High level procedure toepasbaar op gelijk welke optimalisatie-probleem

Bevat volgende elementen:
- Initiële oplossing bepalen (at random of d.m.v. een eenvoudige heuristiek)
- Toepasbaar 'lokaal zoek'-principe: huidige oplossing vervangen door 'betere' oplossing in de buurt
- Toelaten om af en toe toch naar een 'slechtere' buurt te gaan
- Gebaseerd op een analogie uit de fysica, de biologie of de ethologie
- Parameters sturen de duur van de heuristiek en de kwaliteit van de gevonden oplossing

Eigenlijk een simpel principe, I.P.V. dat we altijd gaan naar waar het verbeterd (of waar minder straling is) gaan we af en toe ook naar de kant waar het verslechterd om te zien of het daarna niet toch beter wordt.

Dat is meta-heuristiek, af en toe de slechte kant op gaan om te zien of het daarachter beter is.

#### Voorbeeld
**Simulated Annealing**
- Is gebasseerd op het afkoelen van een metaal
>[!note] DEZE GEBRUIKEN WIJ
## Simulated annealing
Hier zullen we dus lokaal zoeken waar het verbeterd en af en toe ook een stap zetten in de verkeerde richting om te zien of het daarna verbeterd.

Komt van koelproces metalen
- Atomen in materiaal bewegen, maar hoe lager de temperatuur hoe minder
- Materiaal opwarmen om in juiste vorm te krijgen, nadien terug afkoelen
	- Afkoeling te snel -> Onzuiverheden
	- Afkoeling geleidelijk -> Sterkere kristalstructuren
- Eindtoestand wordt dus geleidelijk aan bereikt

Na iedere "sprong" of stap zal de **beste oplossing** onthouden worden, en hoe meer sprongen we doen in de foute richting hoe kleiner de kans wordt dat we nog foute sprongen zullen maken naargelang het proces vorderd. 

### Code
Ziet er ingewikkeld uit maar is best simpel
```python
InitializeParameters (Temperature t, TemperatureReduction α) 
initialSolution (Solution s)
s* = s  //best found solution
while t > TMIN 
	temperatureIteration = 0
	while temperatureIteration < maxIterations
		s’=SelectNeighbour(s)
		Δ = objectiveFunction(s’) – objectiveFunction(s)
		// objectiveFunction must be minimized
		if (Δ < 0)
		then s = s’
			if objectiveFunction(s’) < objectiveFunction(s*) 
				then s* = s’
			else if atRandom[0,1] < exp(-Δ/T) 
				then s = s’
	end while
	t = α*t 
end while 
return s*
```

Wij gebruiken hiervoor een python package **simanneal**
Helpt ons probleem-specifieke berekeningen te scheiden van Metaheurstiek-specifieke berekeningen:
> [!example] 
> Probleem-specifieke berekeningen
>- **move**: Hoe van een oplossing naar een buur-oplossing te gaan
>- **energy**: Berekent de waarde van de objectieve functie voor een oplossing

>[!Example]
>Metaheuristiek-specifieke berekeningen
>- Opgeven van de annealing parametes, zo niet, worden er default waarden gebruikt
>- Uitvoeren van de simulated annealing historiek

#### Voorbeeld
Rastigin functie - Klassieke case om optimalisatie algoritmen en heuristieken te testen
- Oplossingsruimte (Dit zijn de Constraints (waarbinnen we moeten blijven)): (x, y) met x en y = \[-5.12, 5.12]
- Ddelfunctie (de formule voor het doel, wat we willen bereiken) = f(x) = 20 + x² + 10.cos(2\*pi\*x) + y² - 10 \* cos(2 \* pi \* y)

**Code**: 
```python
from simanneal import Annealer
class RastriginProblem(Annealer): 
	def move(self):
		# x: self.state[0] en y: self.state[1] 
		i = np.random.randint(0, 2)
		self.state[i] += np.random.normal(0, 0.1)
		self.state[i] = np.clip(self.state[i], -5.12, 5.12)
	
	def energy(self):
		sum = 20 + self.state[0]**2 - 10*math.cos(2*math.pi*self.state[0]) + self.state[1]**2 - 10*math.cos(2*math.pi*self.state[1])

	return sum

init_sol = np.random.uniform(-5.12, 5.12, size=2) # initiele [x, y]
rastrigin = RastriginProblem(init_sol) # initiele [x, y] -> self.state
# opgeven van de annealing parameters. Zo niet: default waarden
 rastrigin.anneal()
```
Wat er hier gebeurt is: we erfen over van de simanneal klasse, en moeten telkens 2 klassen zelf definiëren

**Move**: Hier definiëren we welke stappen we moeten zetten
**Energy**: Hier zien we welke formule/test we moeten uitvoeren

Eens deze gemaakt is kunnen we een object van die klasse maken met een initiele oplossing -> een willekeurige startpositie (Hier zien we dat bij *init_sol* nemen we 2 random waarden voor x en y tussen de constraints -5.12 en 5.12)
-> uniform wilt zeggen dat ze allemaal een gelijke kans hebben, elke keuze heeft even veel kans om gekozen te worden. Zo kieze we 2 getallen wat een numpy array zal zijn met een *X* van -5.12 en + 5.12 en *Y* tussen -5.12 en 5.12 

Dan zullen we met `rastigin = RastiginProblem(init_sol)` onze klasse aanroepen en het resultaat bewaren en met `rastigin.anneal()` zullen we de uitkomst te zien krijgen

In de *move* genereren we een random getal tussen 0 en 2 (**2 EXCLUSIEF**) waarmee we at random X of Y zullen verplaatsen (we gebruiken deze random waarde als index van de array x = 0, y = 1) Hier gaan we dan een random "normaal" getal bijtellen (volgens de normaalverdeling, volgt de klokcurve kijk volgende foto) waar het midden dus 0 is en de standaardafwijking 0.1. We gaan dus of kleine negatieve of kleine positieve stapjes zetten. 
![[troubleshooting-timing-circuits-f6.webp]]

De **np.clip** zal kijken of een stap buiten de constraints valt en indien dit zo is zal hij deze gewoon op de grens zetten

in de *energy* gaan we de doelfunctie definieren

**Doelfunctie**: ook wel cost function genoemd is de functie die we proberen minimaliseren of maximaliseren. Deze bepaalt hoe "goed" een bepaalde oplossing is

- Bij *minimalisatie* hoe **lager** de waarde van de doelfunctie, hoe beter de oplossing
- Bij *maximalisatie* hoe **hoger** de waarde, hoe beter.

Deze bepaalt dus eigenlijk waar we naar op zoek zijn en hoe we zullen beoordelen of een oplossing beter of slechter is. 

We kunnen een doelfunctie ook bekijken als een thermometer die aangeeft hoe goed je huidige oplossing is. Simulated annealing gebruikt deze om te beslissen of het een stap wel of niet neemt.
### Default waarden
```python
Tmax = 25000.0 # MAX (start) temperatuur
Tmin = 2.5 # MIN (eind temperatuur)
temperature steps = 50000 # Aantal iteraties
updates = 100 # Aantal updates
```

>[!warning]
>By default zal onze library de energy functie *minimaliseren* wat we niet willen als we bijvoorbeeld naar een maximum zoeken. Hiervoor moeten we het resultaat van de objectieve functie \*-1 doen 

#### Voorbeeld TSP
```python
class TSPProblem(Annealer): 
	def move(self):
		# Swaps two cities in the route
		a = np.random.randint(0, len(self.state))
		b = np.random.randint(0, len(self.state))
		self.state[a], self.state[b] = self.state[b], self.state[a]
	def energy(self):
		# Length of the route without for loop
		from_city = self.state
		# shift the array one position to the right 
		to_city = np.roll(from_city, -1)
		return distance_matrix[from_city, to_city].sum()

initial_state = [0, 4, 1, 3, 2]
tsp = TravellingSalesmanProblem(initial_state)
route, distance = tsp.anneal()
```

**initial_state**: hier kiezen we een random volgorde van de steden
**move**: Hier gaan we random 2 steden wisselen en kijken wat er met de thermometer gebeurt
**enegry**: Hier gaan we de lengte van de hele route berekenen

