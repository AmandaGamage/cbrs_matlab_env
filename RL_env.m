% ===========================
% Load ONNX CNN Model
% ===========================
cnnModel = importONNXNetwork("E:\Msc\Lab\data\fid_data\model_GAN.onnx", 'OutputLayerType', 'classification');

% ===========================
% Define RL Environment
% ===========================
function state = getState(img, cnnModel, channelStatus, SNR, userPriority)
    % Resize input to match CNN model input size
    img = imresize(img, cnnModel.Layers(1).InputSize(1:2));
    
    % Convert grayscale to RGB by replicating the single channel
    if size(img, 3) == 1
        img = repmat(img, [1, 1, 3]);
    end
    
    % Classify spectrogram image
    classIdx = classify(cnnModel, img);  % Get class index (1 to 12)
    
    % Convert incumbent_classes to categorical if classIdx is categorical
    incumbent_classes = categorical([5, 6, 9, 10, 11, 12]);  % These classes contain incumbents
    incumbent_present = ismember(classIdx, incumbent_classes);  % 1 if incumbent, 0 otherwise
    
    % Extract CNN feature embeddings from the fully connected layer
    featureVector = activations(cnnModel, img, 'x_model_layer4_cb_45');  % Use fully connected layer
    
    % Construct the state vector
    state = [double(incumbent_present); featureVector(:); channelStatus(:); SNR; userPriority];
end



function img = generateCBRSSpectrogram()
    % Define frequency bins (e.g., 3.55 GHz to 3.7 GHz)
    freqBins = 50;  % Number of frequency bins
    timeBins = 100; % Number of time slots
    
    % Simulate spectrum occupancy: Primary Users (high power), GAA (low power)
    spectrogramData = rand(freqBins, timeBins); % Random noise base
    
    % Select random rows (frequencies) for incumbent users
    incumbentRows = randi([1 freqBins], 10, 1); % Pick 10 random rows
    spectrogramData(incumbentRows, :) = repmat(0.8 + 0.2*rand(1, timeBins), length(incumbentRows), 1);
    
    % Select random rows (frequencies) for GAA users
    gaaRows = randi([1 freqBins], 5, 1); % Pick 5 random rows
    spectrogramData(gaaRows, :) = repmat(0.4 + 0.1*rand(1, timeBins), length(gaaRows), 1);
    
    % Convert to grayscale image (simulate spectrogram)
    img = uint8(255 * mat2gray(spectrogramData));
    
    % Save image for RL environment
    imwrite(img, "simulated_spectrogram.png");
end

function channelStatus = simulateCBRSChannels()
    % Define 5 CBRS channels
    numChannels = 5;
    
    % Generate random occupancy states (0 = free, 1 = occupied)
    % PAL users occupy channels with 50% probability
    % GAA users randomly occupy remaining channels
    incumbent = randi([0 1], numChannels, 1); % Incumbent users
    PAL = (rand(numChannels, 1) < 0.5); % PAL users
    GAA = (rand(numChannels, 1) < 0.3) & ~incumbent; % GAA users on free channels
    
    % Combine into one channel status vector
    channelStatus = incumbent + PAL + GAA;
end


% ===========================
% Define Action Space
% ===========================
actionSpace = [1, 2, 3, 4]; % 1: Stay, 2: Switch, 3: Reduce Power, 4: Delay Transmission
actInfo = rlFiniteSetSpec(actionSpace);

% ===========================
% Define Reward Function
% ===========================
function reward = getReward(action, incumbent_present)
    if action == 1 && incumbent_present == 1
        reward = -2; % Staying on a busy channel
    elseif action == 2 && incumbent_present == 0
        reward = +2; % Successfully switched to free channel
    elseif action == 3
        reward = +1; % Reduced power, preventing interference
    elseif action == 4
        reward = +1; % Delayed transmission to avoid interference
    else
        reward = -1; % Penalty for poor decision
    end
end

% ===========================
% Implement RL Step Function
% ===========================
function [nextState, reward, isDone, loggedSignals] = stepFunction(action)
    % Load spectrogram image
    img = imread("current_spectrogram.png");
    
    % Load ONNX CNN model
    cnnModel = importONNXNetwork("trainedCNN.onnx", 'OutputLayerType', 'classification');
    
    % Generate random environment conditions
    channelStatus = simulateCBRSChannels(); % Generate dynamic channel occupancy
    SNR = rand();  % Simulated SNR
    userPriority = randi([0,2]);  % 0=GAA, 1=PAL, 2=Incumbent
    
    % Get CNN-based state representation
    state = getState(img, cnnModel, channelStatus, SNR, userPriority);
    
    % Compute reward based on action
    reward = getReward(action, state(1)); % state(1) = incumbent presence
    
    % Simulate environment transition
    nextState = getState(img, cnnModel, channelStatus, SNR, userPriority);
    
    % Define stopping condition (optional)
    isDone = false; % Set to true if termination criteria are met
    loggedSignals = [];
end

generateCBRSSpectrogram(); % Generate a synthetic spectrogram

% ===========================
% Create RL Environment
% ===========================
img = imread("simulated_spectrogram.png"); % Load the generated spectrogram
obsDim = length(getState(img, cnnModel, [0;0;0;0;0], 0, 0)); % State vector size
obsInfo = rlNumericSpec([obsDim,1], 'LowerLimit', -inf, 'UpperLimit', inf);

% Define RL environment
env = rlFunctionEnv(obsInfo, actInfo, @stepFunction);
% ===========================
% Train the RL Agent
% ===========================
% Choose RL Algorithm: PPO (for continuous decision-making) or DQN (for discrete actions)

% PPO Agent (Recommended)
agentOptions = rlPPOAgentOptions('ExperienceHorizon', 100);
agent = rlPPOAgent(rlRepresentation(obsInfo, actInfo, 'ActorCritic'), agentOptions);

% Training
trainOpts = rlTrainingOptions('MaxEpisodes', 1000, 'Verbose', true);
trainingStats = train(agent, env, trainOpts);

% ===========================
% Test the RL Agent
% ===========================
simOpts = rlSimulationOptions('MaxSteps', 50);
sim(agent, env, simOpts);
