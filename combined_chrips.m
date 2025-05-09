clear; clc; close all

%% Global Parameters
fs = 20e6;                        
duration = 250e-6;                
t = (0:1/fs:duration-1/fs)';     

% Spectrogram params
window = hamming(200);
noverlap = 124;
nfft = 256;

% Frequency shift for upper spectrum
fc_shift = 5e6;

%% Lower half: Radar only (no shift)
lower = getChannelSignal('Radar', t, 0, fs);

%% Upper half: Radar + LTE (shifted up)
upper = getChannelSignal('Collision', t, fc_shift, fs);

%% Combine both bands
combined = lower + upper;
combined = combined(:);

%% Plot Spectrogram
figure;
spectrogram(combined, window, noverlap, nfft, fs, 'yaxis');
title('Spectrogram: Lower = Radar | Upper = Radar + LTE');
xlabel('Time (s)');
ylabel('Frequency (MHz)');
ylim([6 12]);
colormap gray;
colorbar;

%% --- Helper Functions ---

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
end

function sig = generateRadar(t)
    chirp_duration = 40e-6;     
    spacing = 45e-6;            
    f0_start = 11e6;            
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
    M = 16; 
    Nsub = 64; % More subcarriers = better frequency resolution
    numSymbols = floor(length(t)/Nsub);
    
    % Generate random QAM symbols
    dataBits = randi([0 M-1], Nsub, numSymbols);
    qamSymbols = qammod(dataBits, M, 'UnitAveragePower', true);
    sigMatrix = zeros(Nsub, numSymbols); % initialize all subcarriers with 0

    % Frequency mapping
    freq_resolution = fs / Nsub;
    freq_vector = (-fs/2 : freq_resolution : fs/2 - freq_resolution);  % centered at 0
    % Find indices for subcarriers within 8–11 MHz range
    target_mask = (freq_vector >= 7e6 & freq_vector <= 13e6);
    subcarrier_indices = find(target_mask);

    % Shift indices to match MATLAB's FFT ordering (1-based)
    subcarrier_indices = mod(subcarrier_indices - 1 + Nsub/2, Nsub) + 1;

    % Insert high power burst only in selected subcarriers
    burstIdx = round(numSymbols / 2);
    burstWidth = 10;
    endIdx = min(burstIdx + burstWidth-1, numSymbols);
    burstPower = 200;

    % Smooth burst window
    w = hann(burstWidth*2+1);
    w = w(burstWidth+1:end); % take second half for fade-in only

    for k = subcarrier_indices
        sigMatrix(k, burstIdx:endIdx) = burstPower * w(1:(endIdx - burstIdx + 1));
    end

    % Optional: add a fixed tone within the burst range (e.g., at 9 MHz)
    tone_freq = 9e6;
    tone_idx = find(abs(freq_vector - tone_freq) == min(abs(freq_vector - tone_freq)), 1);
    tone_idx = mod(tone_idx - 1 + Nsub/2, Nsub) + 1;
    sigMatrix(tone_idx, :) = 4;

    % Add noise
    sigMatrix = sigMatrix + 0.1 * randn(size(sigMatrix));

    % Convert to time domain
    tempSig = ifft(sigMatrix, Nsub);
    sig = reshape(tempSig, 1, []);
    sig = sig(1:min(length(t), length(sig)));
    % Apply Tukey window (cosine-tapered) to smooth edges
    w = tukeywin(length(sig), 0.2)';
    sig = sig .* w;

    if length(sig) < length(t)
        sig(end+1:length(t)) = 0;
    end
    sig = sig(:);
end
