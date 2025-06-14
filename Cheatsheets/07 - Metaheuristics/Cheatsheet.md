```python
class Problem(Annealer): 
	def energy(self):
		result = here_comes_a_formula
		return result
	
	def move(self):
		# Swaps two cities in the route
		a = np.random.randint(0, len(self.state)) # Random value in array
		b = np.random.randint(0, len(self.state)) # Random value in array
		self.state[a], self.state[b] = self.state[b], self.state[a] # In case we want to switch values
		
		i = np.random.randint(0, 2) # Random int between 0 and 1 - used for X and Y values
		self.state[i] += np.random.normal(0, 0.1) # Take a stap in a random direction using the random index
		self.state[i] = np.clip(self.state[i], lowest-constraint, highest-constraint) # In case we use X and Y values
		

initial_state = np.random.uniform(-5.12, 5.12, size=2) # one particular case, where we set constraints between -5.12 and 5.12 and we want a x and y value to check
problem = Problem(initial_state) # start the function with the initial state 
problem.anneal() # start the annealing
```