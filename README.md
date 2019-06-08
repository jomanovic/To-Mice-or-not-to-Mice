# To-Mice-or-not-to-Mice
![](https://github.com/jomanovic/To-Mice-or-not-to-Mice/blob/master/images/MOUSE.png)

This is a Python implementation of Multiple Imputations by Chained Equations (MICE): 
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/

## What are Missing values or NaN's?

- [](http://www.stat.columbia.edu/~gelman/arm/missing.pdf)

## What is Multiple Imputations by Chained Equations?: 

![](https://github.com/jomanovic/To-Mice-or-not-to-Mice/blob/master/images/MICE.jpg)

### Prerequisites

- Pandas 0.24.2
- Scikit-Learn 0.21.2

## How do I use it?

    class MiceImputer(object):
    
      def __init__(self, seed_values = True, seed_strategy="mean", copy=True):
          self.strategy = seed_strategy # seed_strategy in ['mean','median','most_frequent', 'constant']
          self.seed_values = seed_values # seed_values = False initializes missing_values using not_null columns
          self.copy = copy
          self.imp = SimpleImputer(strategy=self.strategy, copy=self.copy)
    
      def fit_transform(self, X, method = 'Linear', iter = 5, verbose = True):
          method = ['Linear', 'Ridge']
          
    m = MiceImputer()
    m.fit_transform(dataset)

