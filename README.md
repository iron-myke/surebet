# surebet
A library to compute rankings for football leagues, and look for winning betting strategies
### Classes
- **League** (league.py) : computes the daily league rankings from matches as csv file
- **Strategy** (strategy.py) : computes metrics for a given strategy, and looks for winning strategies with optimization algorithms.
### Run the scripts in Ananconda prompt:
- Compute features for all leagues: 
```sh
python league.py 
```
-  Compute all strategies:
```sh
python find_strategies.py
```
- Visualize strategies:
```sh
jupyter notebook
```
Then, go to visualize_strategy.ipynb
