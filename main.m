clear ; close all; clc

%% we choose to have 3-layered neural network.
input_layer_size  = 400;  % our inputs are 20 pixel by 20 pixel images.
hidden_layer_size = 25;   % 25 units in our hidden layer.
num_labels = 10;          % 10 output units for each of the digits. 


fprintf('Loading and Visualizing Data ...\n')

load('picture_data.mat');
y=data(:,401);
data(:,401)=[]; X=data;
m = size(X, 1);

sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));
%here, we randomly select 100 samples of our dataset and display them to the user.
fprintf('Program paused. Press enter to continue.\n');
pause;


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
%we choose to initialize the weights with values other than zero to avoid the symmetry problem.

lambda=3;
[J, grad]=nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
checkNNGradients(lambda);% when you're working with Neural letworks, it's really important to numerically check if we've got the right answers.
fprintf('Program paused. Press enter to continue.\n');
pause;

options = optimset('MaxIter', 50);%by 200 iterations, we can achieve an accuracy near 97%. 

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
%now we have trained the NN and have the needed parameteres to predict the new inputs.



Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;


rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end

