# reinforcement
This is a collection of reinforcement learning scripts that were written while following "Deep Reinforcement Learning Hands-On" by Maxim Lapan. 



## Cross Entropy Method
The cross entropy method is pretty simple and works by letting an agent play in the environment. After some time of exploring, we get the best performing ones and train a model with state as input and action as output on those instances. 
Then we let this new trained agent play in the environment many times and train on its best runs. And we repeat this process multiple times. A model is trained on the cart-pole environment using this method in x_entropy.py


## Tabular Learning
This one is a little more complicated, but the main idea is that we can make a table that maps state to action, and using some concepts from dynamic programming (Bellman equation), we can improve our table to optimize our performance. 
This is demonstrated in tabular_learning.py for the frozen lake environment. 
