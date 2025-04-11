% Step 1: Load original CNN
load('cnn_ddpm_resnet.mat'); 
net = trainedNet;            

% Step 2: Prepare binary-labeled CBRS spectrograms
dataFolder = 'E:\Msc\Lab\data\fid_data\original_data'; 
allFolders = dir(dataFolder);
folderNames = {allFolders([allFolders.isdir] & ~startsWith({allFolders.name}, '.')).name};

binaryDataFolder = fullfile(dataFolder, 'binary_labeled');
mkdir(binaryDataFolder);

for i = 1:length(folderNames)
    folder = folderNames{i};
    src = fullfile(dataFolder, folder);
    if contains(lower(folder), 'collision')
        dst = fullfile(binaryDataFolder, 'collision');
    else
        dst = fullfile(binaryDataFolder, 'noncollision');
    end
    mkdir(dst);
    images = dir(fullfile(src, '*.png'));
    for j = 1:length(images)
        copyfile(fullfile(src, images(j).name), fullfile(dst, images(j).name));
    end
end

% Step 3: Load and split
imds = imageDatastore(binaryDataFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[trainImds, valImds] = splitEachLabel(imds, 0.8, 'randomized');

% Step 4: Preprocess (grayscale + resize)
inputSize = [224 224];
trainImds.ReadFcn = @(x) imresize(rgb2gray(imread(x)), inputSize);
valImds.ReadFcn   = @(x) imresize(rgb2gray(imread(x)), inputSize);

% Step 5: Modify network
lgraph = layerGraph(net);

% Replace final 3 layers
numClasses = 2;
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_binary')
    softmaxLayer('Name', 'prob')  % Match existing name!
    WeightedClassificationLayer([1 2], 'output')  % Custom loss
];

lgraph = replaceLayer(lgraph, 'fc_binary', newLayers(1));
lgraph = replaceLayer(lgraph, 'prob', newLayers(2));
lgraph = replaceLayer(lgraph, 'output', newLayers(3));

% Step 6: Define training options
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',8, ...
    'MiniBatchSize',32, ...
    'Shuffle','every-epoch', ...
    'ValidationData',valImds, ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ExecutionEnvironment','auto');

% Step 7: Train
fineTunedNet = trainNetwork(trainImds, lgraph, options);

% Step 8: Save
save('fineTunedCBRSCNN.mat', 'fineTunedNet');
