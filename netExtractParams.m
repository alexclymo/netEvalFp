function netParams = netExtractParams(net)
%netExtractParams Extracts weights, biases, and other info from a shallow 
% feedforward net. Output netParams is used to evaluate the level and first
% derivative of the approximated function using netEvalF and netEvalFp.
%
% Author: Alex Clymo
% Date: 23/06/2025
%
% NOTE: This code builds very heavily on Alessandro Villa's ANNEA code
% available at https://github.com/forket86/ANNEA/
%
% This function assumes default MATLAB preprocessing (mapminmax) is used
% for both inputs and outputs.
%
% Output:
%   netparams - struct with fields:
%       IW, B1       - input-to-hidden weights and biases
%       LW, b2       - hidden-to-output weights and biases
%       xminin, xmaxin, yminin, ymaxin - for input scaling
%       xminout, xmaxout, yminout, ymaxout - for output scaling
%       other fields correspond to more info about the net structure

% Activation functions used in hidden and output layer
netParams.actFnHidden = net.layers{1}.transferFcn;
netParams.actFnOut = net.layers{2}.transferFcn;

% Size of input and output vectors
netParams.inSize = net.inputs{1}.size;
netParams.outSize = net.outputs{2}.size;

% Network weights and biases
netParams.IW = net.IW{1,1};     % Input to hidden layer weights
netParams.B1 = net.b{1};        % Hidden layer biases
netParams.LW = net.LW{2,1};     % Hidden to output layer weights
netParams.b2 = net.b{2};        % Output layer bias

% Input preprocessing (mapminmax scaling parameters)
psin = net.inputs{1}.processSettings{1};
netParams.procFnIn = psin.name;
netParams.ymaxIn = psin.ymax;
netParams.yminIn = psin.ymin;
netParams.xmaxIn = psin.xmax;
netParams.xminIn = psin.xmin;

% Output postprocessing (mapminmax inverse scaling)
psout = net.outputs{2}.processSettings{1};
netParams.procFnOut = psout.name;
netParams.ymaxOut = psout.ymax;
netParams.yminOut = psout.ymin;
netParams.xmaxOut = psout.xmax;
netParams.xminOut = psout.xmin;

% Check network satisfies Matlab defaults. Otherwise, functions netEvalF
% and netEvalFp are not correctly defined and you need to define new
% versions to work with these different options
if ~(strcmp(netParams.actFnHidden,'tansig') && strcmp(netParams.actFnOut,'purelin') && strcmp(netParams.procFnIn,'mapminmax') && strcmp(netParams.procFnOut,'mapminmax'))
    error('Network does not use Matlab default options. Not compatible with netEvalF and netEvalFp')
end

