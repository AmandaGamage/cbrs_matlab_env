%% Step 1: Load pretrained network
load('cnn_ddpm_resnet.mat'); 
net = trainedNet;  % Replace with your model variable name if different

%% Step 2: Convert 12-class folder structure into binary 
sourceDir = 'E:\Msc\Lab\data\fid_data\original_data';
binaryDir = fullfile(sourceDir, 'binary_labeled');

if ~isfolder(binaryDir)
    mkdir(binaryDir);
    folders = dir(sourceDir);
    for i = 1:length(folders)
        name = folders(i).name;
        if folders(i).isdir && ~startsWith(name, '.')
            src = fullfile(sourceDir, name);
            if contains(lower(name), 'collision')
                dst = fullfile(binaryDir, 'collision');
            else
                dst = fullfile(binaryDir, 'noncollision');
            end
            mkdir(dst);
            imgs = dir(fullfile(src, '*.png'));
            for j = 1:length(imgs)
                copyfile(fullfile(src, imgs(j).name), fullfile(dst, imgs(j).name));
            end
        end
    end
end

%% Step 3: Load datastore
imds = imageDatastore(binaryDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainImds, valImds] = splitEachLabel(imds, 0.8, 'randomized');

%% Step 4: Preprocess - grayscale and resize
inputSize = [224 224];
trainImds.ReadFcn = @(x) imresize(rgb2gray(imread(x)), inputSize);
valImds.ReadFcn   = @(x) imresize(rgb2gray(imread(x)), inputSize);

%% Step 5: Update network for binary classification
lgraph = layerGraph(net);
numClasses = 2;

% Replace last layers
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_binary')
    softmaxLayer('Name', 'prob')
    classificationLayer('Name', 'output')
];
lgraph = replaceLayer(lgraph, 'fc_binary', newLayers(1));
lgraph = replaceLayer(lgraph, 'prob', newLayers(2));
lgraph = replaceLayer(lgraph, 'output', newLayers(3));

%% Step 6: Compute class weights
tbl = countEachLabel(trainImds);
weights = max(tbl.Count) ./ tbl.Count;

%% Step 7: Custom classification layer with weights
classNames = categories(trainImds.Labels);
weightedLayer = WeightedClassificationLayer(weights, 'output');
lgraph = replaceLayer(lgraph, 'output', weightedLayer);

%% Step 8: Train network (initial)
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',10, ...
    'MiniBatchSize',32, ...
    'Shuffle','every-epoch', ...
    'ValidationData',valImds, ...
    'Verbose',false, ...
    'Plots','training-progress');

net1 = trainNetwork(trainImds, lgraph, options);

%% Step 9: Hard Negative Mining
[YPred, ~] = classify(net1, valImds);
YTrue = valImds.Labels;

% Find misclassified: false negatives
falseNegIdx = (YPred == classNames(1)) & (YTrue == classNames(2));  % Predicted noncollision, actually collision
files = valImds.Files(falseNegIdx);
hardNegLabels = YTrue(falseNegIdx);

% Create new datastore with hard negatives
hardNegDS = imageDatastore(files);
hardNegDS.Labels = hardNegLabels;
hardNegDS.ReadFcn = valImds.ReadFcn;

% Augment training data
trainImdsAug = imageDatastore([trainImds.Files; hardNegDS.Files]);
trainImdsAug.Labels = [trainImds.Labels; hardNegDS.Labels];
trainImdsAug.ReadFcn = trainImds.ReadFcn;

%% Step 10: Retrain network with hard negatives
netFinal = trainNetwork(trainImdsAug, lgraph, options);

%% Step 11: Save final network
save('fineTunedCBRS_HardNegNet.mat', 'netFinal');
