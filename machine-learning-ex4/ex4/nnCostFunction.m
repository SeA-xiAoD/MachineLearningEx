function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
% Theta1 has size 25 x 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% Theta2 has size 10 x 26

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


EyeMatrix = eye(num_labels);

% compute J

a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(m,1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

tempJ = 0;
for i = 1:m
    for k = 1:num_labels
        sigNow = a3(i,k);
        now_y = EyeMatrix(k,y(i));
        tempJ += -now_y*log(sigNow)-(1-now_y)*log(1-sigNow);
    endfor
endfor

J = tempJ / m;

tempM1 = Theta1(:,2:end).*Theta1(:,2:end);
tempM2 = Theta2(:,2:end).*Theta2(:,2:end);
J += (sum(tempM1(:))+sum(tempM2(:)))*lambda/(2*m);

% end compute J
% ------------------------------Part 2------------------------------------
% compute grad

X = [ones(m,1) X];

D_1 = zeros(size(Theta1));
D_2 = zeros(size(Theta2));

for t=1:m,

	% dummie pass-by-pass
	% forward propag

	a1 = X(t,:); % X already have bias
	z2 = Theta1 * a1';

	a2 = sigmoid(z2);
	a2 = [1 ; a2]; % add bias

	z3 = Theta2 * a2;

	a3 = sigmoid(z3); % final activation layer a3 == h(theta)


	% back propag (god bless me)

	z2=[1; z2]; % bias

	delta_3 = a3 - EyeMatrix(:,y(t)); % y(k) trick - getting columns of t element
	delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2);

	% skipping sigma2(0)
	delta_2 = delta_2(2:end);

	Theta2_grad = Theta2_grad + delta_3 * a2';
	Theta1_grad = Theta1_grad + delta_2 * a1;

end;

% ----------------------------Part 3---------------------------------

Theta1_grad(:,1) = Theta1_grad(:,1) ./ m;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) ./ m + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,1) = Theta2_grad(:,1) ./ m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) ./ m + (lambda/m) * Theta2(:,2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
