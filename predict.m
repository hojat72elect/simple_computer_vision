function p = predict(Theta1, Theta2, X)
m = size(X, 1);%5000
num_labels = size(Theta2, 1);%10

p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
a2=sigmoid(Theta1*X');
a2=[ones(1,size(a2, 2));a2];
h=sigmoid(Theta2*a2);
h=h';
[y,p]= max(h, [], 2);
end