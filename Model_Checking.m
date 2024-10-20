%%
netData = load("C:\mit-bih-arrhythmia-database-1.0.0\trained_net_lr_0.0001.mat");
netTransfer = netData.netTransfer;
%%
% Open a file selection dialog to choose an image
[fileName, filePath] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'}, 'Select an Image');
if isequal(fileName, 0)
    disp('No image selected.');
    return;
end

% Read and display the selected image
selectedImage = imread(fullfile(filePath, fileName));
figure;
imshow(selectedImage);
title('Selected Image');

% Preprocess the image for AlexNet
inputSize = [227, 227];  % Input size for AlexNet
imgResized = imresize(selectedImage, inputSize);

% Ensure the image has 3 channels (RGB)
if size(imgResized, 3) == 1
    imgResized = cat(3, imgResized, imgResized, imgResized);
end


% Classify the image using the trained model
[label, scores] = classify(netTransfer, imgResized);

% Display the classification result
fprintf('Classification Result: %s\n', char(label));
disp('Classification Scores:');
disp(scores);

% Optional: Display classification scores in a bar graph
figure;
bar(scores);
title('Classification Scores');
xlabel('Classes');
ylabel('Score');
xticks(1:numel(scores));
xticklabels((netTransfer.Layers(end).Classes)); % Replace with the correct class labels
xtickangle(45);