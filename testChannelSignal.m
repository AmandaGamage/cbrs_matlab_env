% testCBRS.m — Run signal generator and show spectrogram

fs = 20e6;  % Sampling frequency (20 MHz)
duration = 5e-3;  % Signal duration (5 ms)
t = (0:1/fs:duration-1/fs).';  % Time vector
freqShift = 0;  % Optional shift

% Choose from: 'Empty', 'Radar', 'LTE', 'Collision'
signalType = 'Collision';

% Generate signal
sig = getChannelSignal(signalType, t, freqShift, fs);

% Spectrogram settings (same as training config)
windowLength = 200;
overlap = 124;
nfft = 256;

% Generate spectrogram
[S, F, Tspec] = spectrogram(sig, hamming(windowLength), overlap, nfft, fs, 'yaxis');

% Display grayscale spectrogram
figure;
imagesc(Tspec*1e3, F/1e6, abs(S));  % Time in ms, Freq in MHz
axis xy;
xlabel('Time (ms)');
ylabel('Frequency (MHz)');
title(['Spectrogram - ' signalType]);
colormap gray;
colorbar;
