% Test script approximating a function with vector output with a Neural Net
% and testing the functions netEvalF and netEvalFp which compute the value
% and Jacobian of the approximated function
%
% Author: Alex Clymo
% Repository: https://github.com/alexclymo/netEvalFp
% Date: 23/06/2025
%
% NOTE: This code builds very heavily on Villa & Valaitis's ANNEA code
% available at https://github.com/forket86/ANNEA/
%
% Test approximating a simple function y = f(x) with a NN. Here y is 3 x 1
% vector and x=(x1,x2) is a 2 x 1 vector. We build grids for x1 and x2 and
% evaluate y=f(x) at all nodes to get the true function values y. We then
% train the net using train() to get the net object net, which is the
% approximation to the function f(x). To get predicted values of y for a
% given input x simply call yhat = net(x).
% train() performs supervised learning, training the model to predict the
% y data we give it. The neural net is a shallow neural net with the default
% matlab parameterisation.
% Finally we test netEvalF and netEvalFp (which compute the value and
% Jacobian of the approximated function) to make sure they return the
% correct values.
%
% In this example, we add more elements to the y vector, which increases
% the number of parameters to be estimated. The fit is again visibly off
% with 5 hidden nodes, and gets very good by 15 nodes. 

clear all
close all
clc

%set seed (needed because train() initialise the neural net with a random
%set of near-zero parameters)
rng(42)


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up function to approximate and underlying x and y data

% function to approximate: y = f(x) where x = (2 x 1) vector and y is
% (3 x 1) vector. Function f(x) accepts input of size 2 x Ns where each of 
% the Ns columns is one x value at which to evaluate the function
f = @(x) [x(1,:).^2.*(1 + 0.2*(x(2,:)>1.2).*(x(2,:)-1.2) ); 2*x(2,:).^3 .*(1 + 0.5*(x(1,:).^2>1.1).*(x(1,:).^2-1.1) ) ; 3*x(1,:).*x(2,:)];

%define x grids
x1ss = 1;
x2ss = 1;
Delta1 = 0.8;
Delta2 = 0.5;
N1 = 101;
N2 = 111;
x1grid = x1ss*linspace(1-Delta1,1+Delta1,N1)';
x2grid = x2ss*linspace(1-Delta2,1+Delta2,N2);
%replicate onto (x1,x2) space
x1s = x1grid*ones(1,N2);
x2s = ones(N1,1)*x2grid;
%x data for neural net: each "sample" is a column -> matrix size 2 x Ns
x = [x1s(:),x2s(:)]';
Ns = size(x,2);

%true y value: 1 x Ns vector of true evaluation of f(x) function at each x
y = f(x);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialise neural net

% Set up one layer neural net with Nn neurons
Nn = 15; %number of nodes in the hidden layer
net = feedforwardnet(Nn); %can also use net = fitnet(Nn);

% Number of parameters = Nn*(Nx+Ny+1) + Ny where Nn=number of hidden layer 
% nodes, Nx is dimension of input vector, and Ny dimension of output vector
Nx = size(x,1); %size of x input (2 x 1 vector)
Ny = size(y,1); %size of y output (3 x 1 vector)
Npars = Nn*(Nx + Ny +1) + Ny % number of parameters estimated in the net

% show training window or not
net.trainParam.showWindow = 1;

% Important: choose non-random split of traning and validation data if want
% to reduce randomness across, e.g., iterations of a loop. This keeps the
% default 70% training, 15% validation 15% test split. Can alter the split
% by changing the divideParam ratios below
net.divideFcn = 'divideint';
%net.divideParam.trainRatio =
%net.divideParam.valRatio =
%net.divideParam.testRatio =

% Maximum number of update iterations (epochs) inside the training
% function. Default is 1000
net.trainParam.epochs = 1000;

%print info on what activation functions are used in each layer
for i = 1:numel(net.layers)
    fprintf('Layer %d uses activation function: %s\n', i, net.layers{i}.transferFcn);
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train NN

%call training: net is our trained network, tr contains training diagnostics
[net,tr] = train(net,x,y);

%predicted y values
yhat = net(x);

%Check Approximation error on the grid: mean absolute percentage error
meanApproxErr = mean(abs((yhat-y)./y),'all')

%Check Approximation error on the grid: R squared
SS_res = sum((y - yhat).^2, 'all');
SS_tot = sum((y - mean(y, 'all')).^2, 'all');

R2 = 1 - SS_res / SS_tot

%Check why training stopped
trStop = tr.stop
if strcmp(tr.stop,'Training finished: Reached maximum number of epochs')
    warning('Training stopped because hit max no of epochs. Consider raising net.trainParam.epochs if fit is bad.')
end



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot results


MM = Ny; NN = 5;

figsize = [300,200,1200,650];
figure('position',figsize)

for i = 1:Ny

    % reshape y and yhat to (N1,N2) matrices for plotting
    ys = reshape(y(i,:)',[N1,N2]);
    yhats = reshape(yhat(i,:)',[N1,N2]);



    subplot(MM,NN,1 + (i - 1)*5)
    mesh(x2grid,x1grid,ys)
    title('Data')
    xlabel('x2')
    ylabel('x1')
    zlabel(['y',num2str(i)])

    subplot(MM,NN,2 + (i - 1)*5)
    mesh(x2grid,x1grid,yhats)
    title('Fitted')
    xlabel('x2')
    ylabel('x1')
    zlabel('yhat')

    subplot(MM,NN,3 + (i - 1)*5)
    mesh(x2grid,x1grid,(ys-yhats)./ys)
    title('Error')
    xlabel('x2')
    ylabel('x1')
    zlabel('(y - yhat)/y')

    subplot(MM,NN,4 + (i - 1)*5)
    i2list = [1,round((N2+1)/2),N2];
    plot(x1grid,ys(:,i2list))
    hold on
    plot(x1grid,yhats(:,i2list),'--')
    xlim([min(x1grid),max(x1grid)])
    grid on
    title('Slices')
    xlabel('x1')
    ylabel('y and yhat')
    legend('low x2: data','mid x2: data','high x2: data','low x2: fitted','mid x2: fitted','high x2: fitted','location','northwest')
    
    subplot(MM,NN,5 + (i - 1)*5)
    i1list = [1,round((N1+1)/2),N1];
    plot(x2grid,ys(i1list,:)')
    hold on
    plot(x2grid,yhats(i1list,:)','--')
    xlim([min(x2grid),max(x2grid)])
    grid on
    title('Slices')
    xlabel('x2')
    ylabel('y and yhat')
    legend('low x1: data','mid x1: data','high x1: data','low x1: fitted','mid x1: fitted','high x1: fitted','location','northwest')


end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test manually constructed function netEvalF vs Matlab's true output
% net(x). Should be identical up to machine tolerance

%extract parameters from net
netParams = netExtractParams(net);

%evaluate net at all points x using Matlab function and manual function
y0_nn = net(x);
y0_man = netEvalF(x,netParams);

% Average error in manual function evaluation relative to calling net(x)
Yerr_mean = mean(abs(y0_man(:) - y0_nn(:)))

if max(abs(y0_man - y0_nn)) > 1e-10
    error('Manual evaluation function netEvalF not giving correct output')
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test manually constructed Jacobian function netEvalPf versus a finite
% difference approximation of the Jacobian. These need not be identical
% since the finite difference approximation is only an approximation. But
% they should be close so we check the average difference between them is
% low


% Approximated Jacobian: manually (analytically) computed at a large sample
% of x points
xTest = x(:,(1:10:Ns));
J_man = netEvalFp(xTest,netParams);

% Approximated Jacobian: "true" finite difference at all x points
J_fd = zeros(size(J_man));
delta = 1e-5;
for i = 1:size(xTest,2)
    x0 = xTest(:,i);
    y0 = net(x0);
    y0p = net(x0 + delta*[1,0;0,1]);
    %y0 = netEvalF(x0,netParams);
    %y0p = netEvalF(x0 + delta*[1,0;0,1],netParams);
    J_fd(:,:,i) = (y0p - y0)/delta;
end

% Compute difference between manual and finite difference versions
Jerr = J_man - J_fd;
Jerr = Jerr(:);

% Mean and maximum absolute difference of the Jacobian relative to finite
% difference version
Jerr_mean = mean(abs(Jerr))
Jerr_max = max(abs(Jerr))


