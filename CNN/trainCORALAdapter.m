%% Load Pretrained 12-Class CNN Model
whos('-file', 'fineTunedCBRSCNN.mat')

load('fineTunedCBRSCNN.mat', 'fineTunedNet');  

% Define models
lgraph = layerGraph(fineTunedNet);
analyzeNetwork(lgraph)
featureLayer = 'res5b_relu'; % Or 'avg_pool' if no activation exists
numClasses = 12;

% Freeze early layers
layers = lgraph.Layers;
connections = lgraph.Connections;
for i = 1:find(strcmp({layers.Name}, featureLayer)) - 1
    if isprop(layers(i), 'WeightLearnRateFactor')
        layers(i).WeightLearnRateFactor = 0;
        layers(i).BiasLearnRateFactor = 0;
    end
end

% Replace final classification layers
lgraph = replaceLayer(lgraph, 'fc_binary', ...
    fullyConnectedLayer(numClasses, 'Name', 'fc_adapted'));
lgraph = removeLayers(lgraph, 'output');

net = dlnetwork(lgraph, 'OutputNames', {'fc_adapted', 'res5b_relu'});
net = dlupdate(@gpuArray, net);  % <-- Move all learnables to GPU


%% Load Source Dataset (From Folder)
sourceFolder = 'E:\Msc\Lab\data\fid_data\combined_DDPM';
imdsSource = imageDatastore(sourceFolder, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Preprocess function to resize + normalize
preprocessFcn = @(data) preprocessCBRSImage(data);

% Apply transform to dataset
imdsTransformed = transform(imdsSource, preprocessFcn);

% Convert to a table datastore with both X and Y
labelDs = arrayDatastore(imdsSource.Labels, 'IterationDimension', 1);
dsCombined = combine(imdsTransformed, labelDs);

% Use minibatchqueue
mbqSource = minibatchqueue(dsCombined, ...
    'MiniBatchSize', 32, ...
    'MiniBatchFormat', {'SSCB', 'CB'}, ...
    'MiniBatchFcn', @(x, y) deal(x{1}, onehotencode(categorical(y{1}), 1)));


%% Create Target Data On-The-Fly Using Synthetic CBRS Signals
fs = 20e6; duration = 800e-6;
t = (0:1/fs:duration - 1/fs)';
targetBatchSize = 32;

% Helper to generate one spectrogram image from random synthetic signal
generateSyntheticSpectrogram = @() ...
    generateSyntheticSpectrogramFromCBRS(t, fs);

% Function that returns a batch of target spectrograms
function dlImgs = generateTargetBatch(batchSize, t, fs)
    imgs = zeros(224, 224, 1, batchSize, 'single');
    for i = 1:batchSize
        img = generateSyntheticSpectrogramFromCBRS(t, fs);
        imgs(:, :, 1, i) = single(img);
    end
    dlImgs = dlarray(imgs, 'SSCB');
    dlImgs = gpuArray(dlImgs);
end


%% CORAL Adaptation Training Loop
numEpochs = 8;
learnRate = 1e-4;
coralWeight = 0.1;
trailingAvg = [];
trailingAvgSq = [];

for epoch = 1:numEpochs
    reset(mbqSource);
    while hasdata(mbqSource)
        [Xsrc, Ysrc] = next(mbqSource);
        Xtgt = generateTargetBatch(size(Xsrc, 4), t, fs);
        Xsrc = gpuArray(Xsrc);
        Ysrc = gpuArray(Ysrc);
        Xtgt = gpuArray(Xtgt);
        [gradients, loss, state] = dlfeval(@modelGradients, net, Xsrc, Ysrc, Xtgt, coralWeight);
        if isnan(extractdata(loss))
            warning('NaN loss detected, skipping update');
            continue;
        end

        net.State = state;

        [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
            trailingAvg, trailingAvgSq, epoch, learnRate);
    end

    fprintf("Epoch %d complete. Loss: %.4f\n", epoch, extractdata(loss));
end

save('CORALAdaptedCNN.mat', 'net');

function imgOut = preprocessCBRSImage(img)
    imgOut = imresize(single(rgb2gray(img)), [224 224]);
    imgOut = imgOut / 255;
    imgOut = (imgOut - 0.5) / 0.5;  % Normalize to mean 0, std 1 approximately
    imgOut = reshape(imgOut, 224, 224, 1);
end


function [gradients, loss, state] = modelGradients(net, Xsrc, Ysrc, Xtgt, coralWeight)
    % Source forward
    [dlYsrcPred, Fsrc, state] = forward(net, Xsrc);
    
    % Target forward (no label, just features)
    [~, Ftgt] = forward(net, Xtgt);

    % Classification loss
    lossCls = crossentropy(dlYsrcPred, Ysrc);

    % CORAL loss
    lossCORAL = coralLoss(Fsrc, Ftgt);

    % Total loss
    loss = lossCls + coralWeight * lossCORAL;
    gradients = dlgradient(loss, net.Learnables);
end



function loss = coralLoss(A, B)
    % Reshape to [features x batch]
    A = reshape(A, [], size(A, ndims(A)));
    B = reshape(B, [], size(B, ndims(B)));

    % Mean center
    A = A - mean(A, 2);
    B = B - mean(B, 2);

    % Check batch size
    nA = max(size(A, 2), 2); % at least 2
    nB = max(size(B, 2), 2);

    covA = (A * A') / (nA - 1);
    covB = (B * B') / (nB - 1);

    % CORAL Loss
    loss = sum((covA - covB).^2, 'all') / (4 * size(A, 1)^2);
end


function img = generateSyntheticSpectrogramFromCBRS(t, fs)
    types = {'Empty', 'Radar', 'LTE', 'Collision'};
    sig1 = getChannelSignal(types{randi(4)}, t, 0, fs);
    sig2 = getChannelSignal(types{randi(4)}, t, 5e6, fs);
    x = sig1 + sig2;

    window = hamming(200); noverlap = 124; nfft = 256;
    [s, ~, ~] = spectrogram(x, window, noverlap, nfft, fs);
    spec = log(1 + abs(s));
    img = imresize(mat2gray(spec), [224 224]);
end

function [Y, state, features] = forwardWithFeatures(net, X)
    layerName = 'res5b_relu';  % The layer to extract features from
    features = [];
    
    executionState = net.State;
    tempNet = net;

    for i = 1:numel(net.Layers)
        layer = tempNet.Layers(i);
        layerNameCur = layer.Name;

        [X, executionState] = forward(tempNet.Layers(i), X, executionState);

        % Store features when reaching the desired layer
        if strcmp(layerNameCur, layerName)
            features = X;
        end
    end

    Y = X;
    state = executionState;
end
