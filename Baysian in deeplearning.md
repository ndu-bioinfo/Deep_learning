## Baysian model in deep learning

#### Basic review
1) use prior knowledge
2) choose answer that explains observations the most (reduce entropy)
3) avoid making extra assumptions

#### Some rules

* Chain rule P(X,Y) = P(X|Y)P(Y); P(X,Y,Z) = P(X|Y,Z)P(Y|Z)P(Z) ... This applies to any number of P combines

* Sum rule P(X) = sum[-∞,+∞](P(X,Y)dY)

* Bays theorem P(θ|X) = P(X,θ)/P(X) = P(X|θ)P(θ)/P(X)  
in which, P(θ|X) is the Posterior probability, P(X) is evidence, P(X|θ) is likelihood, and P(θ) is the Prior pabability;

#### Baysian approach to statistics
** Difference between frequentist and Baysian 

1) Baysian consider parameter θ to be random while observation X is fixed; frequentist think the opposite. This is important in ML. In training the model, training data was fixed and parameter was adjusted.  
2) The number of parameters, to a frequentist, shoudld be much less than the data points; to Baysian, any data size can work.
3) Frequentist use maximum likelihood, they try to find the θ that maximize the likelihood. Baysian try to determine the posterier P based on Bays theorem. 

Bays formula is great for classification, and the prior P can be treated as regularizer for the regularization, which solves ill-posed problem or prevents overfitting by adding information to the fitting process.



