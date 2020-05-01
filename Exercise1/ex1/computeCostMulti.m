function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful value

m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
j = 1:m;
J = 1/(2*m) * sum(((theta(1).*X(j,1) + theta(2).*X(j,2) + theta(3).*X(j,3) + theta(4).*X(j,4)) - y(j)).^2);



% =========================================================================

end
