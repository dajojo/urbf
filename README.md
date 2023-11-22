# Univariate Radial Basis Function (URBF)



### Regression examples:
Wind turbine dataset:
    - https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset 
    - But is the timeseries dependent on previous values?
Credit card fraud:
    - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    - Is it widely adopted?
A curated list of datasets:
    - https://huggingface.co/datasets/inria-soda/tabular-benchmark
    - Is there a community?
Regression Benchmark suit:
    - https://github.com/slds-lmu/paper_2023_regression_suite
    - But no popularity on github?
statistical regression on photovoltaic electricity production:
    - https://www.sciencedirect.com/science/article/pii/S0038092X13005239?casa_token=T3hSirS0XFkAAAAA:jDK8DAEF5_nbLS5j8ECSs6SyD2uKpyCOgnNWFLa8uUqrmg0HBW1smVVcUSjZLQ95T3htDcONrQ
    - No curated dataset
Machine Learning Benchmarks and Random Forest Regression
    - https://escholarship.org/uc/item/35x3v9t4
    - Not applicable
Multithreaded Local Learning Regularization Neural Networks for Regression Tasks
    - https://www.researchgate.net/publication/282640533_Multithreaded_Local_Learning_Regularization_Neural_Networks_for_Regression_Tasks/figures?lo=1
    - Has a list of datasets..
Penn Machine Learning Benchmarks
    - https://epistasislab.github.io/pmlb/
    - Large collection of datasets
    - Very promising
    - Leaderboard?


TODO: 
 - [ ] Functions with an unequal probability distribution -> Observe behaviour of mean values
 - [ ] Dropout? Study the effect of dropout! Maybe we can observe improved performance
 - [ ] We need to enforce movement of mean values based on activity in a certain region to improve performance


### Observation

When using mean grad input, the initial grad in the first 20 steps diverges...
- Higher grads indicate higher importance
- Lower grads indicate gaussians with lower importance
  
Note: Neurons at the start might be also low since they might not have a great impact.
There is fluent transition between important and unimportant gaussians...

Split & Merge:
1. Condition
   1. Frequency Domain? A strong Highfrequency signal might suggest unwanted behaviour -> Split!
   2. Use relative distance between mean input grads -> Split for highest and merge for lowest
      1. But be aware: noisy signal -> sum over batches until a relative distance is reached 
      2. Split & Merge
      3. Increase threshold by Factor b
      4. repeat
2. Split & Merge
   1. Which Gaussians are split and which are merged?