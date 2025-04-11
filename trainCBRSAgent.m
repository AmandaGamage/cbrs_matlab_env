% Load pretrained CNN
whos('-file', 'cnn_ddpm_resnet.mat')

load fineTunedCBRS_HardNegNet.mat trainedNet 
cnnModel = trainedNet;

% Create environment
env = rlCBRSEnvironment(cnnModel);

% Create observation and action info
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Create a DQN agent
dnn = [
    featureInputLayer(2, 'Normalization', 'none')
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(numel(actInfo.Elements))
    ];

criticOpts = rlRepresentationOptions( ...
    'LearnRate', 1e-3, ...
    'GradientThreshold', 1, ...
    'UseDevice', 'gpu');  

if canUseGPU
    disp('✅ GPU available and will be used.');
else
    warning('⚠️ GPU not available. Training will fall back to CPU.');
end

critic = rlQValueRepresentation(dnn, obsInfo, actInfo, ...
    'Observation', {'input'}, criticOpts);

agentOpts = rlDQNAgentOptions(...
    'SampleTime', 1, ...
    'UseDoubleDQN', true, ...
    'TargetSmoothFactor', 1e-3, ...
    'ExperienceBufferLength', 1e4, ...
    'DiscountFactor', 0.99, ...
    'MiniBatchSize', 64);

agent = rlDQNAgent(critic, agentOpts);

% Train agent
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 1000, ...
    'MaxStepsPerEpisode', 100, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

trainingStats = train(agent, env, trainOpts);

save(['trainedAgent_' datetime("now", 'yyyymmdd_HHMMSS') '.mat'], 'agent');

evaluateCBRSAgent(agent, env); 
