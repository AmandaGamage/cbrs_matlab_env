% Load your pretrained CNN model
load('cnn_ddpm_resnet.mat', 'trainedNet');
cnnModel = trainedNet;

% Define the types
types = {'Empty', 'Radar', 'LTE', 'Collision'};
fs = 20e6;
duration = 800e-6;
t = (0:1/fs:duration - 1/fs)';
fc_shift = 5e6;

% Load your trained CNN model here if not already in memory
% load('yourCNN.mat'); % Uncomment and adjust if needed

% Preallocate results
results = strings(16, 3);
row = 1;

for i = 1:length(types)
    for j = 1:length(types)
        ch1_type = types{i};
        ch2_type = types{j};

        % Generate signals
        ch1 = getChannelSignal(ch1_type, t, 0, fs);
        ch2 = getChannelSignal(ch2_type, t, fc_shift, fs);
        minLen = min(length(ch1), length(ch2));
        ch1 = ch1(1:minLen);
        ch2 = ch2(1:minLen);
        combined = ch1 + ch2;

        % Generate spectrogram image
        window = hamming(200);
        noverlap = 124;
        nfft = 256;
        [s, ~, ~] = spectrogram(combined, window, noverlap, nfft, fs);
        specImage = abs(s);
        specImage = mat2gray(log(1 + specImage));
        specImage = imresize(specImage, [224, 224]);
        % specImage = repmat(specImage, [1 1 3]); % If CNN expects 3-ch input

        % Predict
        pred = classify(cnnModel, specImage);
        results(row, :) = [ch1_type, ch2_type, string(pred)];
        row = row + 1;
    end
end

% Display results
T = cell2table(cellstr(results), 'VariableNames', {'CH1_Type', 'CH2_Type', 'CNN_Prediction'});
disp(T);

