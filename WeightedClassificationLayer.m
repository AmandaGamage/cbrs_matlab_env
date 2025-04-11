classdef WeightedClassificationLayer < nnet.layer.ClassificationLayer
    properties
        ClassWeights
    end

    methods
        function layer = WeightedClassificationLayer(classWeights, name)
            % Constructor
            layer.Name = name;
            layer.ClassWeights = classWeights;
            layer.Description = 'Weighted cross entropy';
        end

        function loss = forwardLoss(layer, Y, T)
            % Ensure Y and T are dlarray
            if ~isa(Y, 'dlarray')
                Y = dlarray(Y);
            end
            if ~isa(T, 'dlarray')
                T = dlarray(T);
            end

            % Numerical stability: clamp predictions
            eps = 1e-8;
            Y = max(min(Y, 1 - eps), eps);

            % Class weights: [numClasses x 1]
            W = layer.ClassWeights(:);

            % Convert to same format as Y for broadcasting
            W = repmat(W, 1, size(Y,2));

            % Weighted cross entropy
            crossEntropy = -sum(W .* T .* log(Y), 1);  % [1 x batchSize]
            loss = mean(crossEntropy);                % scalar

            % Force scalar dlarray
            if ~isscalar(loss)
                loss = sum(loss(:)) / numel(loss);
            end
        end
    end
end
