function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

temp1 = 0;
for i = [1:m]
    temp1 += y(i,1)*log(sigmoid(X(i,:)*theta)) + (1-y(i,1))*log(1-sigmoid(X(i,:)*theta));
endfor

temp2 = 0;
for i =[1:size(theta)]
    temp2 += theta(i,1).^2;
endfor

J = -1 / m * temp1 + lambda / (2 * m) * temp2;

temp = 0;
for i = [1:m]
    temp += (sigmoid(X(i,:)*theta)-y(i,1))*X(i,1);
endfor
grad(1) = temp / m;

for j = [2:size(theta)]
    temp = 0;
    for i = [1:m]
        temp += (sigmoid(X(i,:)*theta)-y(i,1))*X(i,j);
    endfor
    grad(j) = temp / m + lambda / m * theta(j,1);
endfor


% =============================================================

end
