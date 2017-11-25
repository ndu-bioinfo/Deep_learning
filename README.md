# Deep learning Study guide

This document is my personal study note based on the first 3 courses of Andrew Ng's deep learning series on Coursera, containing lecture notes from 9 weeks' worth of workload. Below is a quick breakdown of the contents:

- Neural network and Deep learning
- Improve Deep neural network
- Structure you Machine Learning project 


### A. Neural network and Deep learning     
     - Week 1
     a) each neuron represent a prediction function - e.g. rectified linear unit (ReLU)
     b) each neuron can be a specific ReLU function that represent a specific independent var in relation to Y
     c) Real Estate - commonly use std NN; Image recognition - CNN; Sequence, audio recognition - RNN; auto-drive - custom complex NN
     d) Structured data - tables; Unstructured data - audio or image
     e) larger scale NN takes advantage of the larger amount of data
     f) sigmoid function has gradient has slope close to 0 in the high and low end, makes the learning very slow; ReLU works much better

Geoffrey Hinton interview - UCSD!! "Invented" back prop

     - Week 2
     a) Binary classification problem 1(Y) and 0(N)
     b) Unroll pixel values to feature vector (include all values row* col* n)
     c) x collections of feature vectors, y labels for collections
     d) Logistic regression for binary problem: Given x, find y_hat = P{y=1|x}  # y=1 given x, 0<=y_hat<=1
     e) One step of backward propagation on a computation graph yields derivative of a final output variable
     f) Vectorization: W * xT, in python z=np.dot(W,x)+b. Transferring for loop into matrix calculation
     g) Broadcasting in python allow computations via vectorization; (1,n) or (m,1) matrix will automatically expand to (m,n)
     h) In python specify matrix structure, do not use rank 1 array (n,), do (n,1)

Pieter Abbeel interview - Deep reinforcement learning

     - Week 3
     a) a[1](i) propagate to the next layer as input; a[2](i) = w[2](i).a[1](i)+b[2]
     b) sigmold g'(z=0)~=1/4, tanh'(z=0)~=1; ReLU g'(z>=0) = 1 g'(z<0) = 0
Ian Goodfellow interview - GAN

     - Week 4
     a) Vectorized calculation can only applied to one iteration in one layer
     b) Parameter shapes: W[i]:(n[i],n[i-1]);      b[i]:(n[i],1);      dW[i]:(n[i],n[i-1]);     db[i]:(n[i],1);      A[i]:(n[i],m);     Z[i]:(n[i],m)
     c) b[i] detention will be expanded during Z calculation due to boardcasting in python
     d) Z[l] = W[l].A[l-1]+b[l]
     e) forward a[l-1]->neuron->a[l]+cached Z[l] include W[l] and b[l];     backward da[i]->neuron->da[l-1] + cached dZ[l] include dW[l] and db[l]
     f) Hyperparameters control parameters, they determine the values final parameters
     
### Basic machine learning concept regardless of network structure
####  Forward and Backward propagation
Forward Propagation:
- You get X
- You compute $\hat{Y} = A = \sigma(w^T X + b) = (a^{(0)}, a^{(1)}, ..., a^{(m-1)}, a^{(m)})$
- You calculate the cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$

Here are the two formulas you will be using: 

$$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T$$
$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})$$

####  Optimization
- You have initialized your parameters.
- You are also able to compute a cost function and its gradient.
- Now, you want to update the parameters using gradient descent.

$w$ and $b$ by minimizing the cost function $J$. For a parameter $\theta$, the update rule is $ \theta = \theta - \alpha \text{ } d\theta$, where $\alpha$ is the learning rate.
