function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

might_point = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

tempError = zeros(8,8);

for i =1:8
    for j = 1:8
        temp_model = svmTrain(X, y, might_point(i), @(x1, x2) gaussianKernel(x1, x2, might_point(j)));
        temp_prediction = svmPredict(temp_model, Xval);
        tempError(i,j) = mean(double(temp_prediction ~= yval));
    endfor
endfor

leastError = tempError(1,1);
tempi = 1;
tempj = 1;
for i = 1:8
    for j = 1:8
        if tempError(i,j) < leastError
            tempi = i;
            tempj = j;
            leastError = tempError(i,j);
        end
    endfor
endfor

C = might_point(tempi);
sigma = might_point(tempj);

% =========================================================================

end
