function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
X = [ones(size(X,1),1) X];
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
k = size(Theta1,1);
disp(size(X))
disp(size(Theta1))
A2 = sigmoid(Theta1(1:k, :)*(X(1:m, :))');
A2 = [ones(1,size(A2,2)); A2];
disp(size(A2));
disp(size(Theta2))
A3 = sigmoid((A2(1:size(A2,1),:))'*(Theta2(1:num_labels, :))');
disp(size(A3))
[M, p] = max(A3,[],2);





% =========================================================================


end
