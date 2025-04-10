function sig = getChannelSignal(type, t, freqShift, fs)
    disp(['getChannelSignal: type = ' type ', t size = ' num2str(length(t))]);

    switch type
        case 'Empty'
            sig = 0.001 * randn(size(t));  % Weak noise

        case 'Radar'
            sig = generateRadar(t);

        case 'LTE'
            sig = generatePAL(t, fs);

        case 'Collision'
            sigRadar = generateRadar(t);
            sigPAL = generatePAL(t, fs);
            sigRadar = 100 * sigRadar;
            minLen = min(length(sigRadar), length(sigPAL));
            sig = 0.5 * (sigRadar(1:minLen) + sigPAL(1:minLen));  % Normalize mix

        otherwise
            sig = zeros(size(t));
    end

    % Apply frequency shift if needed
    if freqShift ~= 0
        sig = sig .* exp(1j * 2 * pi * freqShift * t(1:length(sig)));
    end

    % Trim to length of t
    sig = sig(:);
    if length(sig) > length(t)
        sig = sig(1:length(t));
    elseif length(sig) < length(t)
        sig(end+1:length(t)) = 0;
    end
end

%% Radar Signal Generator (Diagonal streaks)
function sig = generateRadar(t)
    chirp_duration = 500e-6;     
    spacing = 550e-6;          
    f0_start = 12e6;
    f0_end = 8e6;
    amplitude = 2.5;

    sig = zeros(size(t));
    total_time = t(end);
    chirp_start_times = 0:spacing:(total_time - chirp_duration);

    for s = chirp_start_times
        idx = (t >= s) & (t < s + chirp_duration);
        local_t = t(idx) - s;

        if isempty(local_t)
            continue;
        end

        % Generate complex I/Q chirp
        i = amplitude * chirp(local_t, f0_start, chirp_duration, f0_end, 'linear');
        q = amplitude * chirp(local_t, f0_start, chirp_duration, f0_end, 'linear', 90);

        % Assign to signal vector safely
        if length(local_t) == sum(idx)
            sig(idx) = sig(idx) + (i + 1i * q);
        end
    end
end

%% PAL (LTE) Signal Generator (Horizontal bars)
function sig = generatePAL(t, fs)
    Nsub = 64;

    % Power configuration
    tonePower = 200;             % Weaker tones (for horizontal lines)
    burstPower = 10000;           % Very strong LTE burst
    burstDuration = 0.6e-3;      % A bit longer than before

    % Compute burst length
    burstSamples = round(burstDuration * fs);
    numSymbols = floor(burstSamples / Nsub);
    totalSamples = Nsub * numSymbols;

    % --- LTE Burst Signal ---
    sigMatrix = zeros(Nsub, numSymbols);
    activeSubcarriers = 28:37; % ~9–11 MHz

    for k = activeSubcarriers
        % LTE burst filled with strong random QAM-like values
        sigMatrix(k, :) = burstPower * (randn(1, numSymbols) + 1i * randn(1, numSymbols));
    end

    % Time-domain OFDM burst
    burstSig = reshape(ifft(sigMatrix, Nsub, 1), [], 1);

    % --- Place LTE burst at center of signal ---
    sig = zeros(size(t));
    totalSamplesAvailable = length(t);
    burstStartIdx = floor((totalSamplesAvailable - length(burstSig)) / 2);
    burstEndIdx = burstStartIdx + length(burstSig) - 1;

    if burstStartIdx > 0 && burstEndIdx <= totalSamplesAvailable
        sig(burstStartIdx:burstEndIdx) = burstSig;
    end

    % --- Add constant tone at 10 MHz before and after burst ---
    toneFreq = 10e6;
    tone = tonePower * exp(1j * 2 * pi * toneFreq * t);

    toneBefore = (t < t(burstStartIdx));
    toneAfter  = (t > t(burstEndIdx));
    sig(toneBefore) = sig(toneBefore) + tone(toneBefore);
    sig(toneAfter)  = sig(toneAfter) + tone(toneAfter);

    % --- Window for smoother edges (optional) ---
    w = tukeywin(length(sig), 0.3);
    sig = sig .* w(:);
end
