function coralLoss = CORALLossLayer(sourceFeatures, targetFeatures)
    % Assume features: [H W C N]
    s = squeeze(mean(sourceFeatures, [1 2]))'; % [C x N]
    t = squeeze(mean(targetFeatures, [1 2]))';

    sCov = cov(s');
    tCov = cov(t');

    diff = sCov - tCov;
    coralLoss = sum(diff(:).^2) / (4 * size(s, 1)^2);
end
