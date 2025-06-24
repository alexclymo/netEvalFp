function y = netEvalF(x,netParams)
%netEvalF Evaluate the network output y at input x using manually extracted 
% weights, biases, and scaling from netParams. This works only for a net
% created using Matlab's default options: tansig activation in hidden
% layer, linear activation in output layer, and mapminmax scaling of the
% input and output layers.
%
% Author: Alex Clymo
% Repository: https://github.com/alexclymo/netEvalFp
% Date: 23/06/2025
%
% NOTE: This code builds very heavily on Villa & Valaitis's ANNEA code
% available at https://github.com/forket86/ANNEA/
%
%   y = netEvalF(netParams, x)
%
% Inputs:
%   netParams - network details, created by netExtractParams
%   x         - input (original scale, not normalized) of size R x S
%               where R is the length of the input vector and each of the S
%               columns is one x value you want to evaluate the function at
%
% Output:
%   y         - output (original scale, not normalized) of size U x S where
%               U is the length of the output vector and S the number of
%               evaluation points

if size(x,1) ~= netParams.inSize || length(size(x)) > 2
    error('Input x must be of size R x S, where R = netParams.inSize and S is the number of x values being evaluated');
end

% Normalize input (mapminmax)
xnorm = (netParams.ymaxIn-netParams.yminIn) * (x - netParams.xminIn) ./ (netParams.xmaxIn - netParams.xminIn) + netParams.yminIn;

% Hidden layer
z1 = netParams.IW * xnorm + netParams.B1;
a1 = tansig(z1);

% Output layer (linear activation)
ynorm = netParams.LW * a1 + netParams.b2;

% Denormalize output (inverse mapminmax)
y = (ynorm - netParams.yminOut) .* (netParams.xmaxOut - netParams.xminOut)./(netParams.ymaxOut-netParams.yminOut) + netParams.xminOut;

