function simulateCBRS()
    fs = 20e6; % Sampling frequency
    t = 0:1/fs:0.01; % Time vector
    freqShift = 0; % Frequency shift

    % Simulate signals
    types = {'Empty', 'Radar', 'LTE', 'Collision'};
    for i = 1:length(types)
        sig = getChannelSignal(types{i}, t, freqShift, fs);
        plotSpectrogram(sig, fs, types{i});
    end
end

function sig = getChannelSignal(type, t, freqShift, fs)
    disp(['getChannelSignal: type = ' type ', t size = ' num2str(length(t))]);

    switch type
        case 'Empty'
            sig = zeros(size(t));

        case 'Radar'
            sig = generateRadar(t);

        case 'LTE'
            sig = generatePAL(t);

        case 'Collision'
            % Process Collision with more control
            sigRadar = generateRadar(t);
            sigPAL = generatePAL(t);
            minLen = min(length(sigRadar), length(sigPAL));

            % Only take the first minLen elements to prevent huge array creation
            sig = zeros(size(t)); % Initialize
            sig(1:minLen) = sigRadar(1:minLen) + sigPAL(1:minLen);

        otherwise
            sig = zeros(size(t));
    end

    % Ensure length matches
    sig = sig(1:min(length(t), length(sig)));

    if freqShift ~= 0
        sig = sig .* exp(1j * 2 * pi * freqShift * t(1:length(sig)));
    end
end

function sig = generateRadar(t)
    chirp_duration = 20e-6;
    spacing = 300e-6;
    f0_start = 12e6;
    f0_end = 8e6;

    sig = zeros(size(t));
    total_time = t(end);
    chirp_start_times = 0:spacing:(total_time - chirp_duration);

    for s = chirp_start_times
        idx = (t >= s) & (t < s + chirp_duration);
        local_t = t(idx) - s;
        if ~isempty(local_t)
            i = chirp(local_t, f0_start, chirp_duration, f0_end, 'linear');
            q = chirp(local_t, f0_start, chirp_duration, f0_end, 'linear', 90);
            sig(idx) = sig(idx) + (i + 1i * q);
        end
    end
end

function sig = generatePAL(t)
    fs = 20e6;
    Nsub = 64; % Number of subcarriers

    numSymbols = min(floor(length(t) / Nsub), 100); % Limit number of OFDM symbols
    sigMatrix = zeros(Nsub, numSymbols);
    
    burstIdx = round(numSymbols / 2);
    burstWidth = min(8, numSymbols - burstIdx + 1);
    burstPower = 500;

    % Generate the signal
    for k = 1:Nsub
        if mod(k, 8) == 0 % Example condition for target subcarriers
            sigMatrix(k, burstIdx:burstIdx + burstWidth - 1) = burstPower;
        end
    end

    tempSig = ifft(sigMatrix, Nsub, 1); % IFFT to obtain time-domain signal
    sig = reshape(tempSig, [], 1); % Reshape into a column vector
    sig = sig(1:min(length(t), length(sig))); % Trim to match t

    % Apply Tukey window
    w = tukeywin(length(sig), 0.3);
    sig = sig .* w(:);

    % Pad with zeros if needed
    if length(sig) < length(t)
        sig(end + 1:length(t)) = 0;
    end
end

function plotSpectrogram(sig, fs, titleStr)
    figure;
    spectrogram(sig, 256, 250, 256, fs, 'yaxis');
    colormap('gray'); % Set colormap to greyscale
    title(titleStr);
    colorbar;
    axis tight;
end