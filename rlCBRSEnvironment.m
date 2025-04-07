classdef rlCBRSEnvironment < rl.env.MATLABEnvironment
    %% Properties (Custom)
    properties
        fs = 20e6
        duration = 250e-6
        t
        fc_shift = 5e6
        
        % Internal state
        currentSignal
        cnnModel % pretrained CNN
    end

    %% RL Properties
    properties
        % Define action: 0 = Idle, 1 = Use CH1, 2 = Use CH2
        ActionInfo = rlFiniteSetSpec([0 1 2])

        % Observation: 1x2 vector: [CH1 occupied, CH2 occupied]
        ObservationInfo = rlNumericSpec([2 1], 'LowerLimit', 0, 'UpperLimit', 1)
    end

    %% Environment State
    properties
        CurrentObservation
        StepCount = 0
        MaxSteps = 100
    end

    methods
        %% Constructor
        function this = rlCBRSEnvironment(cnnModel)
            this.ObservationInfo.Name = 'channel_state';
            this.ActionInfo.Name = 'channel_selection';
            this.t = (0:1/this.fs:this.duration - 1/this.fs)';
            this.cnnModel = cnnModel;

            % Initialize
            reset(this);
        end

        %% Reset Function
        function InitialObservation = reset(this)
            this.StepCount = 0;
            this.CurrentObservation = this.generateObservation();
            InitialObservation = this.CurrentObservation;
        end

        %% Step Function
        function [NextObs, Reward, IsDone, LoggedSignals] = step(this, Action)
            LoggedSignals = [];
            this.StepCount = this.StepCount + 1;

            ch1_occ = this.CurrentObservation(1);
            ch2_occ = this.CurrentObservation(2);

            % SAS-like policy mask
            mask = ~[ch1_occ, ch2_occ];

            % Reward logic
            switch Action
                case 0 % Idle
                    Reward = -0.1; % slight penalty to encourage usage
                case 1 % Use CH1
                    Reward = mask(1) * 1 - ch1_occ * 1;
                case 2 % Use CH2
                    Reward = mask(2) * 1 - ch2_occ * 1;
                otherwise
                    Reward = -1;
            end

            IsDone = this.StepCount >= this.MaxSteps;
            this.CurrentObservation = this.generateObservation();
            NextObs = this.CurrentObservation;
        end
    end

    %% Helper: Generate New Observation using CNN
    methods
        function obs = generateObservation(this)
            % Random scenario selection for each channel
            types = {'Empty', 'Radar', 'LTE', 'Collision'};
            ch1_type = types{randi(4)};
            ch2_type = types{randi(4)};

            ch1 = getChannelSignal(ch1_type, this.t, 0, this.fs);
            ch2 = getChannelSignal(ch2_type, this.t, this.fc_shift, this.fs);

            % Combine both signals
            combined = ch1 + ch2;

            % Generate spectrogram
            window = hamming(200);
            noverlap = 124;
            nfft = 256;
            [s, ~, ~] = spectrogram(combined, window, noverlap, nfft, this.fs);

            % Convert to image
            specImage = abs(s);
            specImage = mat2gray(log(1 + specImage)); % Normalize and log scale
            specImage = imresize(specImage, [227, 227]); % Match CNN input size

            % Predict incumbent presence with CNN
            pred = classify(this.cnnModel, repmat(specImage, 1, 1, 3));

            % Parse prediction into channel state
            % Assume CNN classifies scenarios like: 'Empty', 'Radar', 'LTE', 'Collision'
            % You can adapt this logic to your own CNN output classes
            obs = zeros(2, 1);
            if contains(char(pred), 'Radar')
                obs(1) = 1;
            end
            if contains(char(pred), 'Collision') || contains(char(pred), 'Radar')
                obs(2) = 1;
            end
        end
    end
end
