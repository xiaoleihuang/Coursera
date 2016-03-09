function [J, grad] = nnCostFunction(nn_params, ...
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

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

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
X = [ones(m,1) X];
tempY1 = repmat(1:num_labels,m,1);
tempY2 = repmat(y,1,num_labels);
newY = tempY1 == tempY2;

a2 = sigmoid(X*Theta1');
a2 = [ones(m,1) a2];

a3 = sigmoid(a2*Theta2');

Theta1_temp = Theta1(:,2:end);
Theta1_temp = Theta1_temp.^2;
Theta2_temp = Theta2(:,2:end);
Theta2_temp = Theta2_temp.^2;
cost = log(a3).*-newY - log(1-a3).*(1-newY);% Because y is a m*num_labels matrix instead of m*1 matrix, so use dot product
J = 1/m*sum(sum(cost(:))) + lambda/(2*m)*(sum(Theta1_temp(:))+sum(Theta2_temp(:)));%lambda*(sum(sum(Theta1.^2))+sum(sum(Theta2.^2)))/(2*m)


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
delta_sum2 = 0;
delta_sum1 = 0;
for num = 1:m
    %Step 1:Perform a feedforward pass (Figure 2), 
    %computing the activations(z(2),a(2),z(3),a(3)) for layers 2 and 3.   
    z2 = X(num,:)*Theta1';
    a2 = sigmoid(z2);
    
    a2 = [ones(1,1) a2];
    a3 = sigmoid(a2*Theta2');
    
    %Step 2: Compute delta for layer 3; ?(3) k = (a(3) k ? yk),
    delta3 = a3 - newY(num,:);
    
    %Step 3: Compute delta for layer 2:
    delta2 = (delta3*Theta2)
    delta2 = delta2(1, 2:end);
    delta2 = delta2.*sigmoidGradient(z2);
    
    %Step 4: Accumulate the gradient; 
    %Note that you should skip or remove ?0(2)
    temp1 = delta2'*X(num,:);
    temp2 = delta3'*a2;
    delta_sum1 = delta_sum1 + temp1;
    delta_sum2 = delta_sum2 + temp2;
    
end

Theta2_grad = delta_sum2/m + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];;
Theta1_grad = delta_sum1/m + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
