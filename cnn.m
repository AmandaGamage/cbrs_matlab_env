% Define model names and test dataset paths
models = ["model_GAN", "model_DDPM", "model_VQ-VAE"];
testDataPaths = ["E:\Msc\Lab\data\fid_data\test_GAN", ...
                 "E:\Msc\Lab\data\fid_data\test_DDPM", ...
                 "E:\Msc\Lab\data\fid_data\test_VQ-VAE"];

for modelIdx = 1:length(models)
    model_name = models(modelIdx);
    testDataPath = testDataPaths(modelIdx);
    
    onnx_model_path = fullfile('E:\Msc\Lab\data\fid_data', model_name + ".onnx");
    class_names_path = fullfile('E:\Msc\Lab\data\fid_data', model_name + "_classes.txt");

    % Load class names from file
    fid = fopen(class_names_path, 'r');
    class_names = textscan(fid, '%s');
    fclose(fid);
    class_names = categorical(class_names{1});  % Convert to categorical array

    % Load ONNX model with class names
    net = importONNXNetwork(onnx_model_path, 'OutputLayerType', 'classification', 'Classes', class_names);
    disp("✅ Loaded ONNX model: " + model_name);

    % Load test data
    testData = imageDatastore(testDataPath, ...
        "IncludeSubfolders", true, ...
        "LabelSource", "foldernames", ...
        "ReadFcn", @preprocessImage);  % Apply preprocessing

    % Get true labels
    trueLabels = testData.Labels;
    predictedLabels = categorical();  % Empty array to store predictions

    % Classify images and store predictions
    while hasdata(testData)
        img = read(testData);  % Read image
        img = reshape(img, [448, 448, 3, 1]);  % Ensure batch dimension

        scores = predict(net, img);  % Get class scores
        [~, predictedIdx] = max(scores, [], 2);  % Get class index
        predictedLabels(end+1) = class_names(predictedIdx);  % Use manually loaded class names
    end

    % Compute accuracy for each class
    uniqueClasses = unique(trueLabels);  % Get list of unique class labels
    classAccuracies = zeros(length(uniqueClasses), 1);

    for i = 1:length(uniqueClasses)
        class = uniqueClasses(i);
        correctPredictions = sum(predictedLabels(trueLabels == class) == class);
        totalSamples = sum(trueLabels == class);
        classAccuracies(i) = (correctPredictions / totalSamples) * 100;
    end

    % Display per-class accuracy
    disp("📊 Class-wise Accuracy for " + model_name + ":");
    for i = 1:length(uniqueClasses)
        disp(string(uniqueClasses(i)) + ": " + string(classAccuracies(i)) + "%");
    end

    disp("------------------------------------------------");
end

% Image preprocessing function
function img = preprocessImage(filename)
    img = imread(filename);
    img = imresize(img, [448, 448]);  % Resize to model's input size
    img = im2single(img);  % Convert to single precision

    % Ensure the image has 3 channels (convert grayscale to RGB)
    if size(img, 3) == 1
        img = repmat(img, [1, 1, 3]);  
    end
    img = img(:,:,1:3);  % Ensure exactly 3 channels

    % Add batch dimension (ONNX models expect 4D input)
    img = reshape(img, [448, 448, 3, 1]);  
end
