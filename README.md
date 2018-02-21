# GeneticAlgo_OpenAIGymCartPole
Genetic Algorithm is so powerful that model learns very fast. Most of the cases it performs perfectly during testing may be due to search space is not that large compared to other environments. If we do not end environment it will hold the pole for forever. I run environment up to 10000-time steps :D . Time limit to run env can be changed as mentioned [here](https://github.com/openai/gym/issues/463). 

![](https://github.com/mymultiverse/GeneticAlgo_OpenAIGymCartPole/blob/master/genetic_cart.gif)

Complete version

[![](https://img.youtube.com/vi/gDMYf4xNOF8/0.jpg)](https://www.youtube.com/watch?v=gDMYf4xNOF8)

Here I will explain about my implementation. Let’s start with a brief overview. From cartpole environment we can get observations, awards for each action input (0/1) given to the environment. These observations feed into an artificial neural network which decides what should be the action for next step as shown in below diagram.
![](https://cdn-images-1.medium.com/max/800/1*M6OzpEJzO_8P90KkMqJj3w.png)

We have four observations and one action so neural network consists four input one out. Initially, weights and biases are randomly selected but tuning is required to get perfect action for each time step. Here genetic algorithm plays an important role for the optimal solution of weights and biases. This algorithm works on the survival of the fittest principle so generation(one set of weights and biases) with thebest performance will survive. Above diagram simplified below showing different weight-bias sets associated with different nodes.
![](https://cdn-images-1.medium.com/max/800/1*jW0p6WN-oNn6vsedxBnRsQ.png)

Initially, this weights and biases randomly selected then for each selection the cartpole environment run and scores stored until game over. Now sets with top scores selected and arranged in a particular sequence similar to DNA for crossover (in our case swapping two sequences random) to generate next sequence. The image below shows that how DNAs with red and gray dots exchange some porting with each other to generate two new DNAs.
![](https://cdn-images-1.medium.com/max/800/1*iTRrs0v6V_AgQeutHNvo9A.png)

Exchange point can be anywhere as it is chosen randomly. After crossover mutation is done which is also a random update of few of the dots as a brown dot and a red dot become blue and black respectively after mutation. These new generated along with parent DNAs(weights-biases) put inside the neural network and again see the scores from each and then loop-over the evolution process of genetic algorithm again until perfect score achieved.





Results from Three Layer Neural Network
![](https://github.com/mymultiverse/GeneticAlgo_OpenAIGymCartPole/blob/master/updated.png)
![](https://github.com/mymultiverse/GeneticAlgo_OpenAIGymCartPole/blob/master/new_result.png)

Results from Two Layer Neural Network
![](https://github.com/mymultiverse/GeneticAlgo_OpenAIGymCartPole/blob/master/score_vs_gen.png)

References:-
OpenAI [Gym](https://gym.openai.com/docs/)

