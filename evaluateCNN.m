% Number of evaluation samples
N = 1000;

% Counters
trueLabels = strings(N,1);
predictedLabels = strings(N,1);

for i = 1:N
    % Generate random types
    types = {'Empty', 'Radar', 'LTE', 'Collision'};
    ch1_type = types{randi(4)};
    ch2_type = types{randi(4)};
    
    % Store true label (combined class name)
    trueLabels(i) = "ch1_" + lower(ch1_type) + "_ch2_" + lower(ch2_type);
    
    % Generate signal
    ch1 = getChannelSignal(ch1_type, env.t, 0, env.fs);
    ch2 = getChannelSignal(ch2_type, env.t, env.fc_shift, env.fs);
    combined = ch1(1:min(end, end)) + ch2(1:min(end, end));
    
    % Spectrogram
    window = hamming(200);
    noverlap = 124;
    nfft = 256;
    [s, ~, ~] = spectrogram(combined, window, noverlap, nfft, env.fs);
    specImage = mat2gray(log(1 + abs(s)));
    specImage = imresize(specImage, [224, 224]);
    
    % Predict
    pred = classify(env.cnnModel, specImage);
    predictedLabels(i) = string(pred);
end
% === Step 1: Identify False Negatives ===
falseNegIdx = find(contains(trueLabels, "collision") & predictedLabels ~= "Collision");

% === Step 2: Regenerate and Plot Misclassified Spectrograms ===
numToShow = min(12, length(falseNegIdx));  % Show up to 12 for compact grid
figure('Name','False Negatives - Misclassified Collisions');

for k = 1:numToShow
    idx = falseNegIdx(k);
    
    % Recreate signal using the same random seed pattern
    rng(idx);  % Ensure reproducibility (optional)
    ch1_type = extractBetween(trueLabels(idx), "ch1_", "_ch2");
    ch2_type = extractAfter(trueLabels(idx), "ch2_");
    
    ch1 = getChannelSignal(upperFirst(ch1_type{1}), env.t, 0, env.fs);
    ch2 = getChannelSignal(upperFirst(ch2_type{1}), env.t, env.fc_shift, env.fs);
    combined = ch1 + ch2;

    % Spectrogram
    window = hamming(200);
    noverlap = 124;
    nfft = 256;
    [s, ~, ~] = spectrogram(combined, window, noverlap, nfft, env.fs);
    specImage = mat2gray(log(1 + abs(s)));
    specImage = imresize(specImage, [224, 224]);

    % Plot
    subplot(3, 4, k);
    imshow(specImage);
    title(sprintf('True: %s\nPred: %s', trueLabels(idx), predictedLabels(idx)), ...
          'FontSize', 8, 'Interpreter', 'none');
end

sgtitle('False Negatives (True = Collision, Predicted = Noncollision)', 'FontWeight', 'bold');

% Convert true labels into binary: contains "collision" or not
trueBinary = contains(trueLabels, 'collision');
predBinary = predictedLabels == "Collision";

% Accuracy = correct predictions / total
isCorrect = trueBinary == predBinary;
falsePos = predictedLabels == "Collision" & ~contains(trueLabels, "collision");
falseNeg = predictedLabels ~= "Collision" & contains(trueLabels, "collision");

accuracy = sum(isCorrect) / N * 100;
fprintf('CNN Accuracy on CBRS environment samples: %.2f%%\n', accuracy);

% Optional: confusion matrix
figure;
confusionchart(contains(trueLabels, 'collision'), predictedLabels == "Collision", ...
    'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Binary CNN Confusion Matrix (Collision Detection)');

function out = upperFirst(str)
    out = lower(str);
    if ~isempty(out)
        out(1) = upper(out(1));
    end
end
