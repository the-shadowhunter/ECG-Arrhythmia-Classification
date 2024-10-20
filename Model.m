% Define the path to your complete dataset
datasetPath = "C:\mit-bih-arrhythmia-database-1.0.0\Data_Set";  % change this path to the path in which you are storing Data Set 

% Load the full dataset with imageDatastore
fullDataset = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Define split ratios
trainRatio = 0.7;
valRatio = 0.15;
testRatio = 0.15;

% Split the dataset
[trainImages, valTestImages] = splitEachLabel(fullDataset, trainRatio, 'randomized');
[valImages, testImages] = splitEachLabel(valTestImages, valRatio / (valRatio + testRatio), 'randomized');
% Define destination folders for train, val, and test sets
trainFolder = "E:\Train_Set";  %change this folder path to the path in which you want to save the train images
valFolder = "E:\Validation_Set"; %change this folder path to the path in which you want to save validation images
testFolder = "E:\Test_Set"; %change this folder path to the path in which you want to save test images

% Create the folders and subfolders for each category
categories = unique(fullDataset.Labels);  % Get unique categories/labels

% Create folders for each dataset and label category
for i = 1:length(categories)
    mkdir(fullfile(trainFolder, char(categories(i))));
    mkdir(fullfile(valFolder, char(categories(i))));
    mkdir(fullfile(testFolder, char(categories(i))));
end

% Helper function to copy files to their respective folders
function copyFilesToFolder(imds, destinationFolder)
    % Loop through each file and copy it to the appropriate folder
    for i = 1:numel(imds.Files)
        [~, fileName, ext] = fileparts(imds.Files{i});  % Get file name and extension
        label = char(imds.Labels(i));  % Get the label/category for the image
        
        % Define the target directory
        targetFolder = fullfile(destinationFolder, label);
        
        % Copy the image to the appropriate folder
        copyfile(imds.Files{i}, fullfile(targetFolder, [fileName, ext]));
    end
end

% Copy images to their respective folders
copyFilesToFolder(trainImages, trainFolder);
copyFilesToFolder(valImages, valFolder);
copyFilesToFolder(testImages, testFolder);

disp('Images have been successfully split and saved.');
%%

% Define the desired image size
inputSize = [227 227 3];

% Define image augmentation techniques
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-5, 5]);        % Random rotation between -10 to 10 degrees
    % 'RandXTranslation', [-5, 5], ...    % Random horizontal translation
    % 'RandYTranslation', [-5, 5], ...    % Random vertical translation
    % 'RandScale', [0.9, 1.1]);             % Random scaling between 90% to 110%

% Resize and augment training images
TrainImagesResized = augmentedImageDatastore(inputSize(1:2), trainImages, 'DataAugmentation', imageAugmenter);
% Resize validation and test images without augmentation
ValidationImagesResized = augmentedImageDatastore(inputSize(1:2), valImages);
TestImagesResized = augmentedImageDatastore(inputSize(1:2), testImages);
%%

% Load the pre-trained AlexNet
net = alexnet;
analyzeNetwork(net)
% Extract layers except the last 3 (fully connected, softmax, classification)
layersTransfer = net.Layers(1:end-3);

%% Freeze the convolutional layers by setting their LearnRateFactor to 0 {Not necessarily needed, try it yourself}
% for i = 1:length(layersTransfer)
%     % Only set for convolutional layers (Convolution2DLayer) or fully connected layers
%     if isa(layersTransfer(i), 'nnet.cnn.layer.Convolution2DLayer') 
%         layersTransfer(i).WeightLearnRateFactor = 0;
%         layersTransfer(i).BiasLearnRateFactor = 0;
%     end
% end
%%
% Define the number of classes
numClasses = 5;

% Add custom layers for the new classification task
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses)
    dropoutLayer(0.4)  % Add a dropout layer for regularization 
    softmaxLayer
    classificationLayer
];

% Set the training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...              % Smaller batch size to avoid overfitting
    'MaxEpochs', 32, ...
    'InitialLearnRate', 1e-4, ...         % Lower learning rate for fine-tuning
    'Shuffle', 'every-epoch', ...
    'ValidationData', ValidationImagesResized, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network with augmented data
netTransfer = trainNetwork(TrainImagesResized, layers, options);
%%
% Classify the test images
YPred = classify(netTransfer, TestImagesResized);
YValidation = testImages.Labels;
%%
% Get unique classes
classes = unique(YValidation);

% Initialize metrics storage
accuracy = 0;
precision = zeros(length(classes), 1);
recall = zeros(length(classes), 1);
specificity = zeros(length(classes), 1);
f1Score = zeros(length(classes), 1);

% Loop through each class to calculate metrics
for i = 1:length(classes)
    currentClass = classes(i);
    
    % True Positives: correct predictions for the current class
    TP = sum((YPred == currentClass) & (YValidation == currentClass));
    
    % True Negatives: correct predictions for other classes
    TN = sum((YPred ~= currentClass) & (YValidation ~= currentClass));
    
    % False Positives: wrong predictions as the current class
    FP = sum((YPred == currentClass) & (YValidation ~= currentClass));
    
    % False Negatives: wrong predictions as another class
    FN = sum((YPred ~= currentClass) & (YValidation == currentClass));
    
    % Calculate class-specific metrics
    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    specificity(i) = TN / (TN + FP);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% Overall Accuracy
accuracy = sum(YPred == YValidation) / numel(YValidation) * 100;

% Display the results
fprintf('Overall Accuracy: %.2f%%\n', accuracy);
for i = 1:length(classes)
    fprintf('Class %s - Precision: %.2f%%, Recall (Sensitivity): %.2f%%, Specificity: %.2f%%, F1 Score: %.2f%%\n', ...
        classes(i), precision(i) * 100, recall(i) * 100, specificity(i) * 100, f1Score(i) * 100);
end

% Plot the confusion matrix
plotconfusion(YValidation, YPred);

%%
save('C:\mit-bih-arrhythmia-database-1.0.0\git_hub.mat', 'netTransfer');
