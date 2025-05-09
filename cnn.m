function trainMultiClassCNN(datasetPath, modelSavePath)
    % Load all labeled image data
    imds = imageDatastore(datasetPath, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', ...
        'ReadFcn', @readGrayscaleImage);  % Ensure grayscale images

    % Count the number of classes
    classes = categories(imds.Labels);
    numClasses = numel(classes);

    % Balance classes
    tbl = countEachLabel(imds);
    minCount = min(tbl.Count);
    imds = splitEachLabel(imds, minCount, 'randomized');

    % Split into training and validation sets
    [imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

    % Data augmentation
    augmenter = imageDataAugmenter( ...
        'RandRotation', [-10 10], ...
        'RandXTranslation', [-5 5], ...
        'RandYTranslation', [-5 5]);

    % ResNet18 expects 224x224 images
    inputSize = [224 224 1];
    trainDS = augmentedImageDatastore(inputSize, imdsTrain, ...
        'DataAugmentation', augmenter, ...
        'ColorPreprocessing', 'none');
    valDS = augmentedImageDatastore(inputSize, imdsVal, ...
        'ColorPreprocessing', 'none');

    % Load pretrained ResNet-18
    net = resnet18;
    lgraph = layerGraph(net);

    % === Replace input layer ===
    newInputLayer = imageInputLayer([224 224 1], ...
        'Name', 'input', ...
        'Normalization', 'zerocenter');
    lgraph = replaceLayer(lgraph, 'data', newInputLayer);

    % === Modify first conv layer to accept 1 channel ===
    firstConvLayer = lgraph.Layers(2);
    newWeights = mean(firstConvLayer.Weights, 3); % Average over RGB channels
    newWeights = reshape(newWeights, size(newWeights,1), size(newWeights,2), 1, []);
    newConvLayer = convolution2dLayer(firstConvLayer.FilterSize, ...
        firstConvLayer.NumFilters, ...
        'Stride', firstConvLayer.Stride, ...
        'Padding', firstConvLayer.PaddingSize, ...
        'Weights', newWeights, ...
        'Bias', firstConvLayer.Bias, ...
        'Name', firstConvLayer.Name);
    lgraph = replaceLayer(lgraph, firstConvLayer.Name, newConvLayer);

    % === Replace final layers for 12-class classification ===
    lgraph = replaceLayer(lgraph, 'fc1000', ...
        fullyConnectedLayer(numClasses, 'Name', 'fc_multiclass'));
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', ...
        classificationLayer('Name', 'output'));

    % Training options
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 1e-4, ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', valDS, ...
        'ValidationFrequency', 30, ...
        'ExecutionEnvironment', 'gpu', ...
        'Verbose', false, ...
        'Plots', 'training-progress');

    % Train the network
    trainedNet = trainNetwork(trainDS, lgraph, options);

    % Save the model
    save(modelSavePath, 'trainedNet', 'classes');
end

% === main script: cnn.m ===
datasetPaths = {
    'E:\Msc\Lab\data\fid_data\combined_DDPM'
    'E:\Msc\Lab\data\fid_data\combined_GAN'
    'E:\Msc\Lab\data\fid_data\combined_VQ-VAE'
};

modelNames = {
    'cnn_ddpm_resnet'
    'cnn_gan_resnet'
    'cnn_vqvae_resnet'
};

for i = 1:length(datasetPaths)
    trainMultiClassCNN(datasetPaths{i}, modelNames{i});
end

function img = readGrayscaleImage(filename)
    img = imread(filename);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = im2single(img);  % Normalize to single precision [0,1]
end
