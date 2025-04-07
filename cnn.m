function trainCollisionCNN(datasetPath, modelSavePath)
    % Folders that represent "collision" class
    collisionFolders = [
        "ch1_empty_ch2_collision", ...
        "ch1_collision_ch2_empty", ...
        "ch1_collision_ch2_secondary", ...
        "ch1_primary_ch2_collision"
    ];

    % Load all labeled image data
    imds = imageDatastore(datasetPath, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');

    % Binary relabeling: Collision vs NonCollision
    isCollision = ismember(imds.Labels, collisionFolders);
    imds.Labels = categorical(isCollision, [0 1], {'NonCollision', 'Collision'});

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

    % ResNet18 expects 224x224 RGB images
    inputSize = [224 224 3];
    trainDS = augmentedImageDatastore(inputSize, imdsTrain, ...
        'DataAugmentation', augmenter, ...
        'ColorPreprocessing', 'gray2rgb');
    valDS = augmentedImageDatastore(inputSize, imdsVal, ...
        'ColorPreprocessing', 'gray2rgb');

    % Load pretrained ResNet-18
    net = resnet18;
    lgraph = layerGraph(net);

    % Modify final layers for binary classification
    lgraph = replaceLayer(lgraph, 'fc1000', fullyConnectedLayer(2, 'Name', 'fc_binary'));
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', classificationLayer('Name', 'output'));

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
    save(modelSavePath, 'trainedNet');
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
    trainCollisionCNN(datasetPaths{i}, modelNames{i});
end
