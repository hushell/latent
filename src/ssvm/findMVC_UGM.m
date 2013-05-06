function [X_wo margin] = findMVC_UGM(w, id)
    % finds the most violated constraint on example id under the current w
    % 1st output: The constraint corresponding to that labeling = (Groud Truth Feature - Worst Offending feature)
    % 2nd output: the margin you want to enforce for this constraint.
    % use global variable to communicate with @findMVC_BinaryLinearSVM without
    % passing the data, which can be very expensive.
    
    global labels2train;
    global XnodeTr;
    global XedgeTr;
    global nodeMapTr;
    global edgeMapTr;
    global edgeStructTr;
    
    y = labels2train{id}(:);

    [nodePot,edgePot] = UGM_CRF_makePotentials(w,XnodeTr,XedgeTr,nodeMapTr,edgeMapTr,edgeStructTr,id);
    y_w = UGM_Decode_ICM(nodePot,edgePot,edgeStructTr);
    
    psi_t = featureCB(id, y);
    psi_w = featureCB(id, y_w);
    
    X_wo = psi_t - psi_w;
    margin = lossCB(y, y_w);
end

function delta = lossCB(y, ybar)
% hamming loss
    delta = sum(double(y ~= ybar));
end

function psi = featureCB(id, y)
    % use global variable to communicate with @findMVC_BinaryLinearSVM without
    % passing the data, which can be very expensive.
    lambda1 = 0.85;
    lambda2 = 1.2;
    global XnodeTr;
    global XedgeTr;
    global edgeStructTr;
    nNodeFeatures = size(XnodeTr,2);
    nEdgeFeatures = size(XedgeTr,2);
    %nStates = max(y);
    edgeEnds = edgeStructTr.edgeEnds; % for DEBUG, use global variable better
    
    %psi = zeros(nStates*(nNodeFeatures+nEdgeFeatures), 1);
    psi = zeros(nNodeFeatures+nEdgeFeatures, 1);
    
    psi(1:nNodeFeatures,1) = lambda1 * sum(XnodeTr(id,:,y==1),3);
    %psi(1:nNodeFeatures,1) = sum(XnodeTr(id,:,y==2),3);
    
    psi(nNodeFeatures+1,1) = ...
        sum(XedgeTr(id,1,y(edgeEnds(:,1)) ~= y(edgeEnds(:,2))),3);
    psi(nNodeFeatures+2,1) = ...
        lambda2 * sum(XedgeTr(id,2,y(edgeEnds(:,1)) == y(edgeEnds(:,2))),3);
%     psi(nNodeFeatures+2,1) = ...
%         sum(XedgeTr(id,2,(y(edgeEnds(:,1)) == y(edgeEnds(:,2))) & (y(edgeEnds(:,1)) == 1)),3);
%     psi(nNodeFeatures+3,1) = ...
%         sum(XedgeTr(id,3,(y(edgeEnds(:,1)) == y(edgeEnds(:,2))) & (y(edgeEnds(:,1)) == 2)),3);
end
