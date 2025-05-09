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

%% Step 2: Generate Primary (Radar) Signal
primarySignal = zeros(size(t));
for i = 1:numPulses
    pulseStartIdx = round((i-1) * pulseRepetitionInterval * fs) + 1;
    pulseEndIdx = min(round(pulseStartIdx + pulseWidth * fs), length(t));
    pulseTime = t(pulseStartIdx:pulseEndIdx) - t(pulseStartIdx);
    frequency = f_start + (f_end - f_start) * (i / numPulses); % Linear hopping
    primarySignal(pulseStartIdx:pulseEndIdx) = 10 * sin(2 * pi * frequency * pulseTime); % Strong amplitude
end

%% Step 3: Generate Secondary (GAA/PAL) Signal
M = 16; % 16-QAM
Nsub = 32; 
numSymbols = floor(length(t)/Nsub);

dataBits = randi([0 M-1], Nsub, numSymbols);
qamSymbols = qammod(dataBits, M, 'UnitAveragePower', true);
secondarySignal = ifft(qamSymbols, Nsub);

burstTimeIdx = round(numSymbols / 2);
burstFreqIdx = round(Nsub / 2) + 1;
burstPower = 100; 
burstWidth = 5;
secondarySignal(burstFreqIdx, burstTimeIdx:burstTimeIdx+burstWidth) = burstPower * log2(burstPower) * abs(burstPower); 

toneFreqIdx = round(Nsub / 4);
secondarySignal(toneFreqIdx, :) = 5;

noisePower = 0.1;
secondarySignal = secondarySignal + noisePower * randn(size(secondarySignal));

%% Step 4: Generate Background (if needed)
Nsub = 64;
ofdmSymbols = qammod(randi([0 M-1], Nsub, 1), M);
backgroundSignal = ifft(ofdmSymbols, Nsub);
Nrep = floor(length(t) / Nsub);
backgroundSignal = repmat(backgroundSignal, [1, Nrep]);

%% Step 5: Plot Spectrograms
figure;
subplot(3,1,1);
window = hamming(24);
noverlap = 12;
nfft = 1024;
[~, F, T, P] = spectrogram(primarySignal, window, noverlap, nfft, fs, 'yaxis');
imagesc(T, F/1e6, 10*log10(abs(P)));
colormap(flipud(gray)); axis xy; clim([-80 0]);
ylabel('Frequency (MHz)');
xlabel('Time (s)');
title('Spectrogram of Primary (Radar) User');
colorbar;

subplot(3,1,2);
spectrogram(secondarySignal(:), 64, 50, 128, fs, 'yaxis'); 
colormap gray;
title('Spectrogram of Secondary User');

subplot(3,1,3);
spectrogram(backgroundSignal(:), 128, 120, 128, fs, 'yaxis');
colormap gray;
title('Spectrogram of Background User');
