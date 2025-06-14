K-means clustering is een unsupervised machin learning algoritme, wilt zeggen dat wij geen 'juiste' antwoorden gaan meegeven

Hiermee gaan we clusters aanmaken, belangrijk dat we dit enkel kunnen indien er 2 features zijn, vanaf meer werken we met meerdere dimensies en dat kunnen we niet

## Parameters
- **n_clusters**
	Default = 8
	Het aantal clusters dat we willen vormen, hoe meer hoe meer kans op overfitting 
- **init**
	kunnen we de clusters 'slimmer' mee maken, zal slimmere punten kiezen om dan zo minder iteraties te moeten doen
- **max_iter**
	Max aantal iteraties default = 300
- **tol**
	wanneer hij stopt met zijn iteraties, default is 1 tot de -4e wilt zeggen dat als ieder punt minder beweegt dan dat dat hij de clusters als klaar zal zien
- **n_init**
	Hoeveel keer het algoritme zal runnen, default is 10 dus by default zal het algoritme 10 keer runnen met default 300 iteraties
- **algoritm**
	Gekozen algoritme, wij doen default = Ilkan
- **random_state** 
	Is de seeding, wij gaan vaak 42 gebruiken

