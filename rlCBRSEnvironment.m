classdef rlCBRSEnvironment < rl.env.MATLABEnvironment
    properties
        fs = 20e6
        duration = 800e-6
        t
        fc_shift = 5e6
        currentSignal
        cnnModel
        classNames % Now assumes all class names like 'ch1_empty_ch2_primary'
    end

    properties
        CurrentObservation
        StepCount = 0
        MaxSteps = 100
        LastAction = 0
        VisualizationEnabled = true
        LastPrediction = '';
    end

    methods
        function this = rlCBRSEnvironment(cnnModel)
            obsInfo = rlNumericSpec([2 1], 'LowerLimit', 0, 'UpperLimit', 3); % now 0-3
            obsInfo.Name = 'channel_state';
            actInfo = rlFiniteSetSpec([0 1 2]);
            actInfo.Name = 'channel_selection';

            this = this@rl.env.MATLABEnvironment(obsInfo, actInfo);

            this.t = (0:1/this.fs:this.duration - 1/this.fs)';
            this.cnnModel = cnnModel;
            reset(this);
        end

        function InitialObservation = reset(this)
            this.StepCount = 0;
            this.CurrentObservation = this.generateObservation();
            InitialObservation = this.CurrentObservation;
        end

        function [NextObs, Reward, IsDone, LoggedSignals] = step(this, Action)
            LoggedSignals = [];
            this.StepCount = this.StepCount + 1;
            this.LastAction = Action;

            ch1_state = this.CurrentObservation(1);
            ch2_state = this.CurrentObservation(2);

            Reward = -0.1; % default small penalty for doing nothing useful

            switch Action
                case 0 % Idle
                    if all((this.CurrentObservation == 0)) % Both channels empty
                        Reward = -0.1; % Slight penalty for idling when channels are free
                    else
                        Reward = 0.2; % Reward for idling if no safe transmission possible
                    end

                case 1 % Use CH1
                    Reward = this.evaluateAction(ch1_state);

                case 2 % Use CH2
                    Reward = this.evaluateAction(ch2_state);
            end

            IsDone = this.StepCount >= this.MaxSteps;
            this.CurrentObservation = this.generateObservation();
            NextObs = this.CurrentObservation;

            if this.VisualizationEnabled
                plotEnvironment(this);
            end
        end

        function reward = evaluateAction(~, state)
            % 0 = Empty, 1 = Primary, 2 = Collision, 3 = Secondary
            switch state
                case 0 % Empty
                    reward = 1.0; % ✅ Good
                case 1 % Primary
                    reward = -2.0; % ❌ Big penalty for harming primary
                case 2 % Collision
                    reward = -1.5; % ❌ Penalty for colliding
                case 3 % Secondary
                    reward = 0.5; % Slight positive if sharing with another LTE
                otherwise
                    reward = -1.0; % Unknown behavior
            end
        end
    end

    methods
        function obs = generateObservation(this)
            % Randomly generate types (could be from a generator or CNN output)
            types = {'Empty', 'Primary', 'Secondary', 'Collision'};
            ch1_type = types{randi(4)};
            ch2_type = types{randi(4)};

            % Actually combine real signals here if needed
            ch1 = getChannelSignal(ch1_type, this.t, 0, this.fs);
            ch2 = getChannelSignal(ch2_type, this.t, this.fc_shift, this.fs);
            minLen = min(length(ch1), length(ch2));
            ch1 = ch1(1:minLen);
            ch2 = ch2(1:minLen);
            combined = ch1 + ch2;
            this.currentSignal = combined;

            % Generate spectrogram
            window = hamming(200);
            noverlap = 124;
            nfft = 256;
            [s, ~, ~] = spectrogram(combined, window, noverlap, nfft, this.fs);

            specImage = abs(s);
            specImage = mat2gray(log(1 + specImage));
            specImage = imresize(specImage, [224, 224]);

            % Predict label
            predScores = predict(this.cnnModel, specImage);
            [~, idx] = max(predScores);
            predStr = this.cnnModel.Layers(end).Classes(idx);
            this.LastPrediction = string(predStr);

            disp(['CNN predicted: ', this.LastPrediction]);

            % Decode CNN label into observation
            obs = decodeLabel(this.LastPrediction);
        end

        function plotEnvironment(this)
            figure(999); clf;
            spectrogram(this.currentSignal, hamming(200), 124, 256, this.fs, 'yaxis');
            title(sprintf('Step: %d | Action: %d', this.StepCount, this.LastAction));
            xlabel('Time'); ylabel('Frequency (MHz)');
            ylim([6 12]); colormap jet; colorbar;
            drawnow;
        end
    end
end

function obs = decodeLabel(label)
    % Decode a label like 'ch1_collision_ch2_primary' into numeric states
    obs = zeros(2,1);

    if contains(label, 'ch1_empty')
        obs(1) = 0;
    elseif contains(label, 'ch1_primary')
        obs(1) = 1;
    elseif contains(label, 'ch1_secondary') || contains(label, 'ch1_lte')
        obs(1) = 3;
    elseif contains(label, 'ch1_collision')
        obs(1) = 2;
    end

    if contains(label, 'ch2_empty')
        obs(2) = 0;
    elseif contains(label, 'ch2_primary')
        obs(2) = 1;
    elseif contains(label, 'ch2_secondary') || contains(label, 'ch2_lte')
        obs(2) = 3;
    elseif contains(label, 'ch2_collision')
        obs(2) = 2;
    end
end
