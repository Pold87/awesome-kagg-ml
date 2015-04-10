function [ ll ] = logloss( p, y )
%LOGLOSS This is the multi-class version of the Logarithmic Loss metric. 
%  Each observation is in one class and for each observation, you submit a 
%  predicted probability for each class. The metric is negative the log 
%  likelihood of the model that says each test observation is chosen 
%  independently from a distribution that places the submitted probability 
%  mass on the corresponding class, for each observation.

   [N, M] = size(p);
   
%  The submitted probabilities for a given product are not required to sum 
%  to one because they are rescaled prior to being scored (each row is 
%  divided by the row sum). In order to avoid the extremes of the log 
%  function, predicted probabilities are replaced with 
%  max(min(p, 1-10^-15), 10^-15)

   p = (p ./ repmat(sum(p, 2), 1, 9));
   p = max(min(p, 1 - 1e-15), 1e-15);
   
%  N is the number of observations, M is the number of class labels, log 
%  is the natural logarithm, y(i, j) is 1 if observation i is in class j 
%  and 0 otherwise, and p(i, j) is the predicted probability that 
%  observation i is in class j.

   ll = -N  \ sum(sum(y .* log(p)));

end

