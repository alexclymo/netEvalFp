function J = netEvalFp(x, netParams)
%netEvalFp Evaluate the Jacobian J = dNN(x)/dx of the network at input x
% using manually extracted weights, biases, and scaling from netParams.
%
% Author: Alex Clymo
% Repository: https://github.com/alexclymo/netEvalFp
% Date: 23/06/2025
%
% Inputs:
%   x         - input (original scale) of size R x S where S is no of
%               x vectors to evaluate
%   netParams - network details, created by netExtractParams
%
% NOTE: This code builds very heavily on Villa & Valaitis's ANNEA code
% available at https://github.com/forket86/ANNEA/
%
% Output:
%   J         - Tensor of Jacobian matrices (U x R) of output w.r.t. input
%               of size (U x R x S)

[R, S] = size(x); %R: length of x vector. S: number of evaluation points
U = netParams.outSize; %U: length of y vector
Nn = size(netParams.IW, 1);  % number of hidden units

if R ~= netParams.inSize  || length(size(x)) > 2
    error('Input x must be of size R x S, where R = netParams.inSize and S is the number of x values being evaluated');
end

% Normalize inputs (mapminmax)
xnorm = (netParams.ymaxIn - netParams.yminIn) .* (x - netParams.xminIn) ./ ...
    (netParams.xmaxIn - netParams.xminIn) + netParams.yminIn;  % (R x S)

% Hidden layer activations
z1 = netParams.IW * xnorm + netParams.B1;   % (Nn x S)
a1 = tansig(z1);                             % (Nn x S)
da1_dz1 = 1 - a1.^2;                         % (Nn x S)

% Construct 3D diagonal matrices D: (Nn x Nn x S)
D = repmat(eye(Nn), 1, 1, S);                % identity base
D = D .* reshape(da1_dz1, Nn, 1, S);         % multiply each column into diag

% Prepare fixed weights for batch multiplication
IW = netParams.IW;                          % (Nn x R)
LW = netParams.LW;                          % (U x Nn)

% Expand IW and LW to 3D tensors for pagemtimes
IW3 = repmat(IW, 1, 1, S);                  % (Nn x R x S)
LW3 = repmat(LW, 1, 1, S);                  % (U x Nn x S)

% Chain rule (without scaling): LW * diag(da1_dz1) * IW
temp = pagemtimes(D, IW3);                  % (Nn x R x S)
dYnorm_dXnorm = pagemtimes(LW3, temp);      % (U x R x S)

% Input and output scaling
dxNorm_dx = diag((netParams.ymaxIn - netParams.yminIn) ./ ...
    (netParams.xmaxIn - netParams.xminIn));      % (R x R)
dy_dyNorm = diag((netParams.xmaxOut - netParams.xminOut) ./ ...
    (netParams.ymaxOut - netParams.yminOut));    % (U x U)

% Final Jacobian: dy/dx = dy/dyNorm * dYnorm/dXnorm * dxNorm/dx
% This applies the fixed dxNorm_dx and dy_dyNorm to every slice
J = pagemtimes(dy_dyNorm, pagemtimes(dYnorm_dXnorm, dxNorm_dx));  % (U x R x S)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% OLD NON-VECTORISED VERSION WHICH ONLY TOOK ONE x AT A TIME AS INPUT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function J = netEvalFp(x,netParams)
% %netEvalFp Evaluate the Jacobian J = dNN(x)/dx of the network at input x
% % using manually extracted weights, biases, and scaling from netParams.
% %
% % Author: Alex Clymo
% % Date: 23/06/2025
% %
% % Inputs:
% %   x         - input (original scale) of size R x 1
% %   netParams - network details, created by netExtractParams
% %
% % % NOTE: This code builds very heavily on Alessandro Villa's ANNEA code
% % available at https://github.com/forket86/ANNEA/
% %
% % Output:
% %   J         - Jacobian matrix (U x R) of output w.r.t. input
% 
% % Check input size
% if size(x,1) ~= netParams.inSize || size(x,2) > 1 || length(size(x)) > 2
%     error('Input x must be of size R x S, where R = netParams.inSize and S is the number of x values being evaluated');
% end
% 
% % Normalize input (mapminmax)
% xnorm = (netParams.ymaxIn - netParams.yminIn) .* (x - netParams.xminIn) ./ ...
%     (netParams.xmaxIn - netParams.xminIn) + netParams.yminIn;
% 
% % Forward pass
% z1 = netParams.IW * xnorm + netParams.B1;  % (Nn x 1)
% a1 = tansig(z1);                           % (Nn x 1)
% 
% % Derivative of tansig: f'(z) = 1 - f(z)^2
% da1_dz1 = 1 - a1.^2;                       % (Nn x 1)
% 
% % Compute dy_norm / dx_norm = LW * diag(da1_dz1) * IW
% dyNorm_dxNorm = netParams.LW * diag(da1_dz1) * netParams.IW;  % (U x R)
% 
% % Chain rule: dx_norm/dx (input scaling Jacobian)
% dxNorm_dx = diag((netParams.ymaxIn - netParams.yminIn) ./ ...
%     (netParams.xmaxIn - netParams.xminIn));      % (R x R)
% 
% % Chain rule: dy/dy_norm (output rescaling Jacobian)
% dy_dyNorm = diag((netParams.xmaxOut - netParams.xminOut) ./ ...
%     (netParams.ymaxOut - netParams.yminOut));    % (U x U)
% 
% % Final Jacobian: dy/dx = dy/dyNorm * dyNorm/dxNorm * dxNorm/dx
% J = dy_dyNorm * dyNorm_dxNorm * dxNorm_dx;  % (U x R)