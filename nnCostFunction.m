function [J, grad] = nnCostFunction(nn_params, ...
      input_layer_size, ...
      hidden_layer_size, ...
      num_labels, ...
      X, y, lambda)
  
 
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
      hidden_layer_size, (input_layer_size + 1));
  
  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
      num_labels, (hidden_layer_size + 1));
  
  m = size(X, 1);
  
  J = 0;
  Theta1_grad = zeros(size(Theta1)); %25 x401
  Theta2_grad = zeros(size(Theta2)); %10 x 26
  
  
  X = [ones(m,1), X];  
  
  a1 = X; % 5000 x 401
  
  z2 = a1 * Theta1';  % m x hidden_layer_size == 5000 x 25
  a2 = sigmoid(z2); % m x hidden_layer_size == 5000 x 25
  a2 = [ones(size(a2,1),1), a2]; % Adding 1 as first column in z = (Adding bias unit) % m x (hidden_layer_size + 1) == 5000 x 26
  
  z3 = a2 * Theta2';  % m x num_labels == 5000 x 10
  a3 = sigmoid(z3); % m x num_labels == 5000 x 10
  
  h_x = a3; % m x num_labels == 5000 x 10
  
  
  y_Vec = (1:num_labels)==y; % m x num_labels == 5000 x 10
  
  J = (1/m) * sum(sum((-y_Vec.*log(h_x))-((1-y_Vec).*log(1-h_x))));  %scalar
  
  
  A1 = X; % 5000 x 401
  
  Z2 = A1 * Theta1';  % m x hidden_layer_size == 5000 x 25
  A2 = sigmoid(Z2); % m x hidden_layer_size == 5000 x 25
  A2 = [ones(size(A2,1),1), A2]; % Adding 1 as first column in z = (Adding bias unit) % m x (hidden_layer_size + 1) == 5000 x 26
  
  Z3 = A2 * Theta2';  % m x num_labels == 5000 x 10
  A3 = sigmoid(Z3); % m x num_labels == 5000 x 10
  
  
  y_Vec = (1:num_labels)==y; % m x num_labels == 5000 x 10
  
  DELTA3 = A3 - y_Vec; % 5000 x 10
  DELTA2 = (DELTA3 * Theta2) .* [ones(size(Z2,1),1) sigmoidGradient(Z2)]; % 5000 x 26
  DELTA2 = DELTA2(:,2:end); % 5000 x 25 %Removing delta2 for bias node
  
  Theta1_grad = (1/m) * (DELTA2' * A1); % 25 x 401
  Theta2_grad = (1/m) * (DELTA3' * A2); % 10 x 26
  
 Theta={Theta1; Theta2}; 
 for i = 1:length(Theta),
  J =J+( lambda / 2 / m * sum(sum(Theta{i}(:,2:end) .^ 2)));
  end   

  
 
  %Calculating gradients for the regularization
  Theta1_grad_reg_term = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; % 25 x 401
  Theta2_grad_reg_term = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; % 10 x 26
  
  %Adding regularization term to earlier calculated Theta_grad
  Theta1_grad = Theta1_grad + Theta1_grad_reg_term;
  Theta2_grad = Theta2_grad + Theta2_grad_reg_term;
  
  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];

end