function [gradients, totalLoss, state] = modelGradients(net, Xsrc, Ysrc, Xtgt, coralWeight)
    [YsrcPred, state] = forward(net, Xsrc, 'Outputs', 'softmax');
    srcLoss = crossentropy(YsrcPred, Ysrc);

    % Extract features from last shared conv layer
    srcFeat = activations(net, Xsrc, 'res5b_relu', 'OutputAs', 'channels');
    tgtFeat = activations(net, Xtgt, 'res5b_relu', 'OutputAs', 'channels');

    coralLoss = CORALLoss(srcFeat, tgtFeat);
    totalLoss = srcLoss + coralWeight * coralLoss;

    gradients = dlgradient(totalLoss, net.Learnables);
end
