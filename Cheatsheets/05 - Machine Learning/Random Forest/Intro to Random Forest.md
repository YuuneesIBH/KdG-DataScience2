Wordt nog meer gebruikt dan de decision tree, ipv één boom te maken worden er hier by default 100 gemaakt - wordt ook een *ensemble* genoemd

IPV een model dat voorspelt of iets een kat of hond is doen er nu 100 dat, en kijken we naar welke prediction het meeste voorvalt en die nemen we

We gaan hiervoor datasets 'sub-samplen' stel een trainingset van 1000 nemen we 900 random samples uit en deze zullen voor iedere tree anders zijn waardoor bijna alle data ooit gebruikt wordt voor te testen en te trainen

## Parameters
- **n_estimators**
	Het aantal trees die we in een forest stoppen - hoe meer hoe beter de performance maar ook meer cpu
- **max_depth**
	*Default = None*
	Is hoeveel aftakkingen we willen, hoeveel lagen diep we willen gaan. Als we deze niet instellen gaat hij door tot we enkel pure-leafs hebben
- **min_samples_split**
	Geeft aan hoeveel samples er minimum moeten zijn om te splitsen, als een leaf hieronder is splitsen we niet meer, pure of niet

