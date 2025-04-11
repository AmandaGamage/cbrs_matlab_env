function [X, Y] = preprocessCBRSImage(data, labels)
    inputSize = [224 224];
    X = cellfun(@(x) imresize(im2single(rgb2gray(imread(x))), inputSize), data, 'UniformOutput', false);
    X = cat(4, X{:});
    X = dlarray(X, 'SSCB');

    if ~isempty(labels)
        Y = onehotencode(categorical(labels), 1);
        Y = dlarray(Y, 'CB');
    else
        Y = [];
    end
end
