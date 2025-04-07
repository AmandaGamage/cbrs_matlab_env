%% Display a Single Example Spectrogram for a Given CBRS Class
% Example class: Channel 1 = Radar, Channel 2 = LTE

clear; clc; close all

%% Global Parameters
fs = 20e6;                        % Sampling frequency: 20 MHz
duration = 250e-6;                % Duration: 250 microseconds
t = (0:1/fs:duration-1/fs)';       % Time vector (column vector)

% Spectrogram parameters
window = hamming(200);
noverlap = 124;
nfft = 256;

% Frequency shift for channel 2 to appear in the upper part
fc_shift = 5e6; % 5 MHz shift

%% Generate Signals for Each Channel
% Channel 1: Radar (incumbent) signal
ch1 = getChannelSignal('Radar', t, 0, fs);

% Channel 2: PAL (burst OFDM) signal
ch2 = getChannelSignal('LTE', t, fc_shift, fs);

% Combine channels (they are independent and added in the time domain)
combined = ch1 + ch2;
combined = combined(:); % ensure a 1-D vector

%% Compute and Display Spectrogram
figure;
spectrogram(combined, window, noverlap, nfft, fs, 'yaxis');
title('Spectrogram for Class: CH1 = Radar, CH2 = LTE');
xlabel('Time (s)');
ylabel('Frequency (MHz)');
colormap gray;
colorbar;

%% ----- Helper Functions -----
function sig = getChannelSignal(type, t, freqShift, fs)
    switch type
        case 'Empty'
            % Very low amplitude noise
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
    % Apply frequency shift if needed (multiply by complex exponential)
    if freqShift ~= 0
        sig = sig .* exp(1j*2*pi*freqShift*t);
    end
end

% -----------------------------
% Helper: Radar Chirp Generator
% -----------------------------
function sig = generateRadar(t)
    % Radar parameters
    f0 = 3.55e9;
    bw = 10e6;
    pulse_duration = 100e-6; % Radar pulse does not span whole signal
    sig = zeros(size(t));
    idx = t <= pulse_duration;
    % Generate chirp for the pulse duration only
    i = chirp(t(idx), f0 - bw/2, pulse_duration, f0 + bw/2, 'linear');
    q = chirp(t(idx), f0 - bw/2, pulse_duration, f0 + bw/2, 'linear', 90);
    sig(idx) = i + 1i*q;
end

% -----------------------------
% Helper: PAL (Burst OFDM) Generator
% -----------------------------
function sig = generatePAL(t)
    M = 16; 
    Nsub = 32;
    numSymbols = floor(length(t)/Nsub);
    dataBits = randi([0 M-1], Nsub, numSymbols);
    qamSymbols = qammod(dataBits, M, 'UnitAveragePower', true);
    sigMatrix = ifft(qamSymbols, Nsub);
    
    % High-power burst in the middle of the OFDM frame
    burstIdx = round(numSymbols / 2);
    burstFreqIdx = round(Nsub / 2);
    burstPower = 100;
    burstWidth = 5;
    endIdx = min(burstIdx+burstWidth, numSymbols);
    sigMatrix(burstFreqIdx, burstIdx:endIdx) = burstPower * log2(burstPower) * abs(burstPower);
    
    % Tone at a lower subcarrier
    toneFreqIdx = round(Nsub / 4);
    sigMatrix(toneFreqIdx, :) = 5;
    
    % Add noise
    sigMatrix = sigMatrix + 0.1 * randn(size(sigMatrix));
    
    % Serialize: reshape into a 1-D time domain signal
    tempSig = ifft(sigMatrix, Nsub);
    sig = reshape(tempSig, 1, []);
    sig = sig(1:min(length(t), length(sig)));
    if length(sig) < length(t)
        sig(end+1:length(t)) = 0;
    end
    sig = sig(:);
end
