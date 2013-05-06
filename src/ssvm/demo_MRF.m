%function learn_UGM_SSVM()

clear
close all
addpath(genpath(pwd));

load ./data_hog.mat

% -------------------------------------------------------------------------
% Run SVM struct
% -------------------------------------------------------------------------

% config
[nRows,nCols,nNodeFeatures] = size(featureMatrices{1});
nNodes = nRows*nCols;
nStates = 2;
nInstances = 7;

% HOG features
X = zeros(nInstances,nNodeFeatures,nNodes); 
for i = 1:nInstances
    X(i,:,:) = reshape(featureMatrices{i},[1,nNodeFeatures,nNodes]);
end

%% Make edgeStruct
adj = sparse(nNodes,nNodes);

% Add Down Edges
ind = 1:nNodes;
exclude = sub2ind([nRows nCols],repmat(nRows,[1 nCols]),1:nCols); % No Down edge for last row
ind = setdiff(ind,exclude);
adj(sub2ind([nNodes nNodes],ind,ind+1)) = 1;

% Add Right Edges
ind = 1:nNodes;
exclude = sub2ind([nRows nCols],1:nRows,repmat(nCols,[1 nRows])); % No right edge for last column
ind = setdiff(ind,exclude);
adj(sub2ind([nNodes nNodes],ind,ind+nRows)) = 1;

% Add Up/Left Edges
adj = adj+adj';
edgeStruct = UGM_makeEdgeStruct(adj,nStates);
nEdges = edgeStruct.nEdges;

%% Make Xnode, Xedge, nodeMap, edgeMap
Xnode = zeros(nInstances,nNodeFeatures+1,nNodes);
for i = 1:nInstances
    Xnode(i,1,:) = ones(1,1,nNodes);
    Xnode(i,2:end,:) = X(i,:,:);
end
nNodeFeatures = size(Xnode,2);

% Make nodeMap
nodeMap = zeros(nNodes,nStates,nNodeFeatures,'int32');
for f = 1:nNodeFeatures
    nodeMap(:,1,f) = f; % state = 1 -> FG, state = 2 -> w = 0
    %nodeMap(:,2,f) = f + nNodeFeatures;
end

% Make Xedge
nEdgeFeatures = 2;
nEdges = size(edgeStruct.edgeEnds,1);
Xedge = zeros(nInstances,nEdgeFeatures,nEdges);

for i = 1:nInstances
    for e = 1:nEdges
        %n1 = edgeStruct.edgeEnds(e,1);
        %n2 = edgeStruct.edgeEnds(e,2);
        Xedge(i,1,e) = 1;
        Xedge(i,2,e) = 1;
        %Xedge(i,3,e) = 1;
    end
end
nEdgeFeatures = size(Xedge,2);

% Make edgeMap
f = max(nodeMap(:));
edgeMap = zeros(nStates,nStates,nEdges,nEdgeFeatures,'int32');
for edgeFeat = 1:nEdgeFeatures
    if edgeFeat == 1
        edgeMap(1,2,:,edgeFeat) = f+edgeFeat;
        edgeMap(2,1,:,edgeFeat) = f+edgeFeat;
    elseif edgeFeat == 2
        edgeMap(1,1,:,edgeFeat) = f+edgeFeat;
        edgeMap(2,2,:,edgeFeat) = f+edgeFeat;
%     elseif edgeFeat == 3
%         edgeMap(2,2,:,edgeFeat) = f+edgeFeat;
    end
end

nParams = max([nodeMap(:);edgeMap(:)]);

%% global variables for training
%global patterns2train;
global labels2train ;
%patterns2train = featureMatrices;
labels2train   = truthMatrices;
for i = 1:nInstances
    labels2train{i}(labels2train{i} == -1) = labels2train{i}(labels2train{i} == -1) + 3;
    labels2train{i} = int32(labels2train{i});
end

global XnodeTr;
global XedgeTr;
global nodeMapTr;
global edgeMapTr;
global edgeStructTr;
XnodeTr = Xnode;
XedgeTr = Xedge;
nodeMapTr = nodeMap;
edgeMapTr = edgeMap;
edgeStructTr = edgeStruct;

%%
param.C = 1.0;
param.max_num_iterations = 1000;
param.max_num_constraints =  100;

w_init = zeros(nParams,1);
%w_init = rand(nParams,1);

[w] = trainOnlineSSVM(numel(labels2train), w_init, @findMVC_UGM, param);

%% Evaluate with learned parameters
%AVATOL_PATH = '/scratch/working/AVATOL/spines/avatol/nematocyst';
AVATOL_PATH = '/Users/zooey/working/AVATOL/spineprojects/avatol/nematocyst';
PATCH_SIZE = 32;
TRAINING_IMAGES_DIR = strcat(AVATOL_PATH, '/data/spines/images/training/');
trainingDir = dir(TRAINING_IMAGES_DIR);
confusionMatrices = cell(nInstances,1);
imgNames = cell(nInstances,1);
cnt = 1;
for iDir = 1:length(trainingDir)
    if ~trainingDir(iDir).isdir
        % get filename and read in image
        fileName = trainingDir(iDir).name;
        fullPath = fullfile(TRAINING_IMAGES_DIR, fileName);
        [filePath, fileName, fileExt] = fileparts(fullPath);
        
        if ~strcmp(fileExt, '.jpg')
            continue
        end
        fprintf('%s...\n', fullPath);
        imgNames{cnt} = fullPath;
        cnt = cnt + 1;
    end
end

fprintf('ICM Decoding with estimated parameters...\n');
figure;
for i = 1:nInstances
    [nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);
    yDecode = UGM_Decode_ICM(nodePot,edgePot,edgeStruct);
    fprintf('ID = %d, logPot = %f\n', i, UGM_LogConfigurationPotential(yDecode,nodePot,edgePot,edgeStruct.edgeEnds));
%     imagesc(reshape(yDecode,nRows,nCols));
%     colormap gray
%     title('ICM Decoding with pseudo-likelihood parameters');
    inferMat = reshape(yDecode,nRows,nCols);
    inferMat(inferMat == 2) = -1;
     
    img = imread(imgNames{i});
    img = rgb2gray(img);
    [img, ~, ~] = resize_image(img, PATCH_SIZE, PATCH_SIZE);
        
    %inferenceMatrix = post_inference_cleanup(inferenceMatrix);
    visualize_inference(img, inferMat, PATCH_SIZE);
    %truthMatrix = truthMatrices{i};
    %confusionMatrices{i} = evaluate_inference(inferMat, truthMatrix);
    
    fprintf('(paused)\n');
    pause
end
