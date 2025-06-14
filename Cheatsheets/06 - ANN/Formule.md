Als we een klein neuraal netwerk zelf zouden willen berekenen kan dit best simpel
![[Pasted image 20250612174337.png]]
We kennen de integration en activation function
![[Pasted image 20250612174613.png]]

En de activation function gebruikt de output layer sigmoid

Stel we hebben input van vorige nodes \[1,1]
En weights \[0.3, -0.1]
en bias -0.7

Dan kunnen we zelf berekenen met
```python
node1 = np.array([1,1])
weights = np.array([0.3, -0.1])
bias = -0.7

# De integration function
integration = np.sum(node1 * weights) + bias

# Activation function
activation = 1 / (1 * math.exp(-integration))

# tonen
print(integration, activation)
```