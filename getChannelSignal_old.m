function sig = getChannelSignal(type, t, freqShift, fs)
    switch type
        case 'Empty'
            sig = 0.001 * randn(size(t));
        case 'Radar'
            sig = generateRadar(t);
        case 'LTE'
            sig = generatePAL(t);
        case 'Collision'
            sig = generateRadar(t) + generatePAL(t);
        otherwise
            sig = zeros(size(t));
    end

    if freqShift ~= 0
        sig = sig .* exp(1j*2*pi*freqShift*t);
    end
    sig = sig(:);
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
        i = chirp(local_t, f0_start, chirp_duration, f0_end, 'linear');
        q = chirp(local_t, f0_start, chirp_duration, f0_end, 'linear', 90);
        sig(idx) = sig(idx) + (i + 1i*q);
    end
end

function sig = generatePAL(t)
    fs = 20e6;
    Nsub = 64;
    numSymbols = floor(length(t)/Nsub);
    
    sigMatrix = zeros(Nsub, numSymbols);
    freq_resolution = fs / Nsub;
    freq_vector = (-fs/2 : freq_resolution : fs/2 - freq_resolution);
    target_mask = (freq_vector >= 9e6 & freq_vector <= 11e6);
    subcarrier_indices = find(target_mask);
    subcarrier_indices = mod(subcarrier_indices - 1 + Nsub/2, Nsub) + 1;

    burstIdx = round(numSymbols / 2);
    burstWidth = 8;
    endIdx = min(burstIdx + burstWidth-1, numSymbols);
    burstPower = 500;

    for k = subcarrier_indices
        sigMatrix(k, burstIdx:endIdx) = burstPower;
    end

    tempSig = ifft(sigMatrix, Nsub);
    sig = reshape(tempSig, 1, []);
    sig = sig(1:min(length(t), length(sig)));

    % Apply window
    w = tukeywin(length(sig), 0.3)';
    sig = sig .* w;

    if length(sig) < length(t)
        sig(end+1:length(t)) = 0;
    end
    sig = sig(:);
end

fs = 20e6;                        % Sampling rate: 20 MHz
T = 1e-3;                         % Duration: 1 ms
t = 0:1/fs:T-1/fs;                % Time vector

sig = getChannelSignal('Radar', t, 0, fs);  % Generate radar signal
spectrogram(sig, 256, 200, 256, fs, 'yaxis');
title('Radar Spectrogram')
