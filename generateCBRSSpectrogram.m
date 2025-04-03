clc; clear; close all;

%% Step 1: Define Parameters
fs = 1e6; % Sampling frequency (1 MHz)
t_total = 1e-3; % Total time duration (1 ms)
t = 0:1/fs:t_total; % Time vector

pulseWidth = 5e-6; % Shorter pulse width
pulseRepetitionInterval = 50e-6; % Pulse repetition interval
numPulses = floor(t_total / pulseRepetitionInterval);

f_start = 3.55e9; % Start frequency (3.55 GHz)
f_end = 3.65e9; % End frequency (3.65 GHz)

%% Step 2: Generate Incumbent (Radar) User Signal with Frequency Hopping
incumbentSignal = zeros(size(t));

for i = 1:numPulses
    pulseStartIdx = round((i-1) * pulseRepetitionInterval * fs) + 1;
    pulseEndIdx = min(round(pulseStartIdx + pulseWidth * fs), length(t));
    
    pulseTime = t(pulseStartIdx:pulseEndIdx) - t(pulseStartIdx);
    
    % Frequency hopping: Increase step size
    frequency = f_start + (f_end - f_start) * (i / numPulses); % Simple linear hopping
    
    incumbentSignal(pulseStartIdx:pulseEndIdx) = 10 * sin(2 * pi * frequency * pulseTime); % Increase amplitude significantly
end

%% **Step 2: Generate PAL User Signal**
M = 16; % 16-QAM modulation
Nsub = 32; % Number of OFDM subcarriers
numSymbols = floor(length(t)/Nsub); % Ensure integer size

dataBits = randi([0 M-1], Nsub, numSymbols); % Random data
qamSymbols = qammod(dataBits, M, 'UnitAveragePower', true); % 16-QAM modulation
palSignal = ifft(qamSymbols, Nsub); % Apply IFFT to generate OFDM

% **Step 3: Add a High-Power Burst in a Localized Region**
burstTimeIdx = round(numSymbols / 2); % Center symbol-wise
burstFreqIdx = round(Nsub / 2)+1; % Center burst in frequency

burstPower = 100; % Higher power for bright intensity
burstWidth = 5;
palSignal(burstFreqIdx, burstTimeIdx:burstTimeIdx+burstWidth) = burstPower * log2(burstPower) * abs(burstPower); 
%palSignal(burstFreqIdx, burstTimeIdx:burstTimeIdx+burstWidth) = burstPower; 

toneFreqIdx = round(Nsub / 4); % Place at a lower subcarrier
palSignal(toneFreqIdx, :) = 5; % Moderate intensity across all time

% **Step 4: Add Background Noise**
noisePower = 0.1;
palSignal = palSignal + noisePower * randn(size(palSignal)); % Add Gaussian noise

%% **Step 4: Generate GAA User Signal (OFDM Modulated)**
Nsub = 64; % Number of OFDM subcarriers
ofdmSymbols = qammod(randi([0 M-1], Nsub, 1), M); % Generate OFDM symbols
gaaSignal = ifft(ofdmSymbols, Nsub); % Apply IFFT for OFDM
Nrep = floor(length(t) / Nsub); % Ensure an integer replication factor
gaaSignal = repmat(gaaSignal, [1, Nrep]); % Repeat to match time

%% **Step 5: Convert to Spectrograms**
figure;

subplot(3,1,1);
spectrogram(incumbentSignal, hann(128), 120, 128, fs, 'yaxis'); % Apply Hann window
colormap gray % Reverse colormap (darker is lower power)
title('Spectrogram of Incumbent (Radar) User');



subplot(3,1,2);
spectrogram(palSignal(:), 64, 50, 128, fs, 'yaxis'); 
colormap gray; % Convert to black and white
title('Spectrogram of PAL User');

subplot(3,1,3);
spectrogram(gaaSignal(:), 128, 120, 128, fs, 'yaxis');
colormap gray;
title('Spectrogram of GAA User');