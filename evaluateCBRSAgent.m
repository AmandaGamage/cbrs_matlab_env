% Load trained agent
load('trainedDQNVQVAEAgent.mat', 'agent');
trainedAgent = agent;

% Load pretrained CNN model
load('cnn_vqvae_resnet.mat', 'trainedNet');
cnnModel = trainedNet;

% Create evaluation environment
env = rlCBRSEnvironment(cnnModel);
env.VisualizationEnabled = false;

% Evaluation settings
numEpisodes = 50;
totalRewards = zeros(1, numEpisodes);
collisionCounts = zeros(1, numEpisodes);
actionLog = [];
predictionLog = {};

% Run episodes
for ep = 1:numEpisodes
    obs = reset(env);
    done = false;
    episodeReward = 0;
    collisions = 0;
    stepCount = 0;

    while ~done
        stepCount = stepCount + 1;
        act = getAction(trainedAgent, obs);
        act = act{1};  % Extract scalar from cell

        %fprintf("Episode %d, Step %d: Action = %d\n", ep, act);

        [obs, reward, done, ~] = step(env, act);
        episodeReward = episodeReward + reward;
        actionLog = [actionLog; act];
        predictionLog{end+1} = env.LastPrediction;

        if (act == 1 && obs(1) == 1) || (act == 2 && obs(2) == 1)
            collisions = collisions + 1;
        end
    end

    totalRewards(ep) = episodeReward;
    collisionCounts(ep) = collisions;
end

% Plot results
figure;
subplot(2,1,1);
plot(totalRewards, '-o');
xlabel('Episode'); ylabel('Total Reward'); title('Episode Rewards');

subplot(2,1,2);
plot(collisionCounts, '-x');
xlabel('Episode'); ylabel('Collisions'); title('Channel Collision Count');

disp('Mean Reward:'), disp(mean(totalRewards));
disp('Mean Collisions per Episode:'), disp(mean(collisionCounts));

% Optional: Save results
% save('evaluation_results.mat', 'totalRewards', 'collisionCounts', 'actionLog', 'predictionLog');
