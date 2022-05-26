%% Constructing a pretrained detector model to detect data corresponding to 20 classes of objects

doTraining = false;

% Downloaded the pretrained open neural network model 
pretrainedURL = 'https://ssd.mathworks.com/supportfiles/vision/deeplearning/models/yolov2/tiny_yolov2.tar';
pretrainedNetTar = 'yolov2Tiny.tar';
if ~exist(pretrainedNetTar,'file')
    disp('Downloading pretrained network (58 MB)...');
    websave(pretrainedNetTar,pretrainedURL);
end

onnxfiles = untar(pretrainedNetTar);
pretrainedNet = 'tiny_yolov2/Model.onnx';

lgraph = importONNXLayers(pretrainedNet,'ImportWeights',true);

onnxAnchors = [1,1; 3,4; 6,11; 9,5; 16,10];

% Loaded data corresponding to all of the attributes of the pretrained
% model
inputSize = lgraph.Layers(1,1).InputSize(1:2);
lastActivationSize = [13,13];
upScaleFactor = inputSize./lastActivationSize;
anchorBoxesTmp = upScaleFactor.* onnxAnchors;
anchorBoxes = [anchorBoxesTmp(:,2),anchorBoxesTmp(:,1)];

weights = lgraph.Layers(end,1).Weights;
bias = lgraph.Layers(end,1).Bias;
layerName = lgraph.Layers(end,1).Name;

% Rearranged the weights and biases of the last layer of the model so that
% the attributes are arranged as per the yolo v2 model
numAnchorBoxes = size(onnxAnchors,1);
[modWeights,modBias] = rearrangeONNXWeights(weights,bias,numAnchorBoxes);

% Replaced the new layer obtained with the old one
filterSize = size(modWeights,[1 2]);
numFilters = size(modWeights,4);
modConvolution8 = convolution2dLayer(filterSize,numFilters,...
    'Name',layerName,'Bias',modBias,'Weights',modWeights);
lgraph = replaceLayer(lgraph,'convolution8',modConvolution8);

classNames = tinyYOLOv2Classes;

% Added the transform and the output layer to the yolo model so obtained
layersToAdd = [
    yolov2TransformLayer(numAnchorBoxes,'Name','yolov2Transform');
    yolov2OutputLayer(anchorBoxes,'Classes',classNames,'Name','yolov2Output');
    ];


% Connected and assembled all the layers together and removed the old layer
lgraph = addLayers(lgraph, layersToAdd);
lgraph = connectLayers(lgraph,layerName,'yolov2Transform');

yoloScaleLayerIdx = find(...
    arrayfun( @(x)isa(x,'nnet.onnx.layer.ElementwiseAffineLayer'), ...
    lgraph.Layers));

if ~isempty(yoloScaleLayerIdx)
    for i = 1:size(yoloScaleLayerIdx,1)
        layerNames {i} = lgraph.Layers(yoloScaleLayerIdx(i,1),1).Name;
    end
    lgraph = removeLayers(lgraph,layerNames);
    lgraph = connectLayers(lgraph,'Input_image','convolution');
end

net = assembleNetwork(lgraph)

% Declared an object for the yolo pretrain detector
yolov2Detector = yolov2ObjectDetector(net)


%% Creating a new model and loading the new training data to it


% The groundtruth data for the new training images
data = load('objectDatasetGroundTruth.mat');
objectDataset = data.objectDataset;

% Display few rows of the data set.
objectDataset(1:4,:)

% Add the fullpath to the local vehicle data folder.
objectDataset.imageFilename = fullfile(pwd,objectDataset.imageFilename);

%% Dividing the data to the train, test and validation sets

rng(0);
shuffledIndices = randperm(height(objectDataset));
idx = floor(0.6 * length(shuffledIndices) );

trainingIdx = 1:idx;
trainingDataTbl = objectDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = objectDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = objectDataset(shuffledIndices(testIdx),:);

%% Storing all the data in the datastore objects so that it can be accessed
% easily
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,2:end));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,2:end));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,2:end));

trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

% Reading the training data and plotting the box on top of it obtained by
% the boxlabel datastore object
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
% figure
% imshow(annotatedImage)

inputSize = [224 224 3];
numClasses = width(objectDataset)-1;

%% Transforming the size of the training data so as to obtain different
% anchor boxes and then finding the corresponding IoU
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)

%% Defining a new pretrained resnet50 network and replacing its layers 

featureExtractionNetwork = resnet50;

featureLayer = 'activation_40_relu';

lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

% Augmenting training data with different sizes/shapes and colours
augmentedTrainingData = transform(trainingData,@augmentData);

% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
% figure
% montage(augmentedData,'BorderSize',10)

preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
preprocessedValidationData = transform(validationData,@(data)preprocessData(data,inputSize));

data = read(preprocessedTrainingData);

I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
% figure
% imshow(annotatedImage)


%% Training the new network on the basis of the pretrained one

options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20, ... 
        'CheckpointPath',tempdir, ...
        'ValidationData',preprocessedValidationData);


if doTraining       
    % Train the YOLO v2 detector.
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
else
    detector = yolov2Detector;
end


%% Trying to detect a few of the images

I = imread('C:\Users\caast\OneDrive\Documents\MATLAB\vehicleImages\image_00190.jpg');
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

I = imread('C:\Users\caast\OneDrive\Documents\MATLAB\vehicleImages\image_00196.jpg');
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

I = imread('C:\Users\caast\OneDrive\Documents\MATLAB\vehicleImages\image_00296.jpg');
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

I = imread('C:\Users\caast\OneDrive\Documents\MATLAB\vehicleImages\image_00297.jpg');
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

I = imread('C:\Users\caast\OneDrive\Documents\MATLAB\vehicleImages\image_00304.jpg');
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)


%% Plotting the accuracy chart of the model

% PLEASE COMMENT "error(message('vision:ObjectDetector:undefinedLabelInDetectionTable', i, msg));
%" LINE IN THE evaluationInputValidation.m FILE PRESENT IN THE LOCATION- 
%The internal function can be found in:
%MATLAB\R2018b\toolbox\vision\vision\+vision\+internal\+detector IN THE C
%DRIVE

preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));

detectionResults = detect(detector, preprocessedTestData);

[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);

recallv = recall{1};
precisionv = precision{1};
ap =  ap(1);

figure
plot(recallv,precisionv)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))
