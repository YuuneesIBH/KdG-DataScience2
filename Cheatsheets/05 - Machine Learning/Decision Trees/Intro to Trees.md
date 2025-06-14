## Decision Trees
Gebruikt verschillende algoritmen, een voorbeeld hiervan gebruikt het *gini-getal* ligt altijd tussen 0 en 0.5
- Hoe dichter bij 0 -> Hoe meer er zijn van eenzelfde klasse
	Wordt ook **pure-leaf** genoemd 
	*Voorbeeld* we willen mannen of vrouwen voorspellen, stel 1 leaf is alleen mannen dan zal dat een gini van 0 hebben wat een pure-leaf is
- Hoe dichter bij 0.5 hoe eerlijker verdeeld
	Stel Origineel 100 mannen en 100 vrouwen, na een aantal keer aftakken een leaf met 20 mannen 20 vrouwen heeft een gini van 0.5

## Parameters
- **criterion** 
	Meet de qualiteit van een splitsing, wij gebruiken gini maar kan ook met entropy zijn
- **max_depth**
	*Default = None*
	Is hoeveel aftakkingen we willen, hoeveel lagen diep we willen gaan. Als we deze niet instellen gaat hij door tot we enkel pure-leafs hebben
- **min_samples_split**
	Geeft aan hoeveel samples er minimum moeten zijn om te splitsen, als een leaf hieronder is splitsen we niet meer, pure of niet
- **min_samples_leaf**
	Hoeveel samples er min in een leaf moeten zitten om een leaf te zijn. te kleine leaves kan voor overfitting zorgen
- **max_features**
	max aantal features die we gaan consideren wanneer we splitten (stel we hebben een dataframe met uur, dag, week, type_zon, type_wind en we willen maar 3 dingen trainen dan kunnen we de max zetten)