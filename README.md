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
 - [x] Functions with an unequal probability distribution -> Observe behaviour of mean values
 - [x] Dropout? Study the effect of dropout! Maybe we can observe improved performance
 - [x] We need to enforce movement of mean values based on activity in a certain region to improve performance
 - [ ] Find Bug which freezes the mean and var... split and merge??
 - [ ] Decide on Model size!
 - [ ] Construct experiments (All Learning Methods with: 0.1, 0.01, 0.001, 0.0001)
   - [x] 1. SVR
   - [x] 2. Lin
   - [x] 3. GradBoost
   - [x] 4. MLP 
   - [x] 5. Vanilla URBF Equal init (urbf_lr x10)
   - [x] 6. Vanilla URBF Spektral init (urbf_lr x10)
   - [x] 7. URBF Split and Merge
   - [x] 8. URBF Adaptive range
   - [ ] 9. URBF dynamic architecture


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



#### Wiki
`register_full_backward_hook`:
1. `grad_outputs`: 
   - These are the gradients with respect to the output of the module.
   - Mathematically, if you consider a module `M` which takes an input `x` and produces an output `y`: `y = M(x)`, then during the backward pass, `grad_outputs` is the gradient of the loss `L` with respect to `y`. This is denoted as d`L`/d`y` .
   - -> `grad_outputs` tells you how much a change in the output of the module will affect the change in the loss.

2. `grad_inputs`:
   - These are the gradients with respect to the input of the module.
   - Continuing with the same module `M`, `grad_inputs` would be the gradient of the loss `L` with respect to the input `x`. This is denoted as d`L`/d`x`.
   - This represents how much changing the input of the module will affect the loss. It's what is backpropagated to the previous layers in a neural network.

`tensor.grad`:
 - By using the grad attribute of a param tensor we can investigate the gradient of a specific param such as mean or var
