function [X_wo margin] = findMVC_BinaryLinearSVM(w, id)
    % finds the most violated constraint on example id under the current w
    % 1st output: The constraint corresponding to that labeling = (Groud Truth Feature - Worst Offending feature)
    % 2nd output: the margin you want to enforce for this constraint.
    % use global variable to communicate with @findMVC_BinaryLinearSVM without
    % passing the data, which can be very expensive.
%     global labels2train;
%     truth = labels2train{id};
%     y_wo = -truth;
%     score = lossCB(truth, y_wo) + dot(w, featureCB(id, y_wo) - featureCB(id, truth));
%     if score <0
%         y_wo = truth;
%     end
%     X_wo = featureCB(id, truth) -  featureCB(id, y_wo);
%     margin = lossCB(truth, y_wo);
    
    global patterns2train;
    global labels2train;
    global edges;
    [M, N] = size(patterns2train{1});
    
    truth = labels2train{id}(:);
    
    % graph cut
    [nodeWeights, edgeWeights] = helper_unstack_weights(w, M, N, edges);
    [y_wo, ~] = inference_crf_GCO(patterns2train{id}, nodeWeights, edgeWeights);
    y_wo = y_wo(:);
    
    X_wo = featureCB(id, truth) -  featureCB(id, y_wo);
    margin = lossCB(truth, y_wo);
end

function delta = lossCB(y, ybar)
%     delta = double(y ~= ybar) ;
    delta = sum(double(y ~= ybar));
end

function psi = featureCB(id, y)
    % use global variable to communicate with @findMVC_BinaryLinearSVM without
    % passing the data, which can be very expensive.
    global patterns2train;
    global edges;
    x = patterns2train{id};
%     psi = y*x;
    I_y = helper_indicator(y, edges);
    psi = [x(:); 1*I_y];
end
