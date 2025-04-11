% Load pretrained 12-class model
load('fineTunedCBRS_HardNegNet.mat'); % assume variable name: trainedNet

% Extract feature layers from ResNet
lgraph = layerGraph(trainedNet);
featureLayer = 'res5b_relu'; % adjust depending on your ResNet
numClasses = 12;
outputLayerName = lgraph.Layers(end).Name;


% Freeze earlier layers
layers = lgraph.Layers;
connections = lgraph.Connections;
for i = 1:find(strcmp({layers.Name}, featureLayer)) - 1
    if isprop(layers(i), 'WeightLearnRateFactor')
        layers(i).WeightLearnRateFactor = 0;
        layers(i).BiasLearnRateFactor = 0;
    end
end

% Modify classification head
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_adapted')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];
lgraph = replaceLayer(lgraph, 'fc_binary', newLayers(1)); 
lgraph = replaceLayer(lgraph, 'output', newLayers(3));
lgraph = removeLayers(lgraph, outputLayerName);

net = dlnetwork(lgraph);

% Load source and target datastores
sourceFolder = 'E:\Msc\Lab\data\fid_data\original_data';
targetFolder = 'E:\Msc\Lab\data\fid_data\target_env_data';

imdsSource = imageDatastore(sourceFolder, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

imdsTarget = imageDatastore(targetFolder, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames'); % labels ignored

% Training parameters
mbqSource = minibatchqueue(imdsSource, ...
    'MiniBatchSize', 32, ...
    'MiniBatchFcn', @(x,y) preprocessCBRSImage(x,y), ...
    'MiniBatchFormat', {'SSCB','CB'});

mbqTarget = minibatchqueue(imdsTarget, ...
    'MiniBatchSize', 32, ...
    'MiniBatchFcn', @(x,y) preprocessCBRSImage(x,[]), ...
    'MiniBatchFormat', {'SSCB'});

% Training loop
numEpochs = 8;
learnRate = 1e-4;
trailingAvg = [];
trailingAvgSq = [];
coralWeight = 0.1;

for epoch = 1:numEpochs
    reset(mbqSource);
    reset(mbqTarget);

    while hasdata(mbqSource) && hasdata(mbqTarget)
        [Xsrc, Ysrc] = next(mbqSource);
        Xtgt = next(mbqTarget);

        [gradients, loss, state] = dlfeval(@modelGradients, net, Xsrc, Ysrc, Xtgt, coralWeight);
        net.State = state;

        [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
            trailingAvg, trailingAvgSq, epoch, learnRate);
    end

    disp("Epoch " + epoch + " complete. Loss: " + extractdata(loss));
end

save('CORALAdaptedCNN.mat', 'net');
