dataset_file = fullfile("Training Data");
categories = ["Afzarah", "Amirah", "Amrita", "ODell", "Omar"];
imds = imageDatastore(fullfile(dataset_file,categories),"LabelSource","foldernames");
%count no of images of each person
tbl = countEachLabel(imds);
%Identify label with min number of images
minCountImages = min(tbl{:,2});
%Update dataset to have 10 randomly chosen images for each label
splitEachLabel(imds, minCountImages, "randomize");
%Load ResNet-50
net = resnet50();
%Visualize architecture of ResNet-50
% figure
% plot(net);
% title("Architecture of ResNet-50");
% Resize figure to understand ResNet-50 Architecture
% set(gca,"YLim",[150 170]);
%Inspect Layer 1 to understand input layer properties for ResNet-50
net.Layers(1);
%Split Train and Test Dataset with a ratio of 70:30
[trainingSet,testSet] = splitEachLabel(imds,0.7,'randomized');
%Data pre-processing
%Set image resolution (224 224 3)
imageResolution = net.Layers(1).InputSize;
% augmentedImageDatastore to ensure dataset fulfils requirements of ResNet-50
augmentedTrainingSet = augmentedImageDatastore(imageResolution, trainingSet, ...
    'ColorPreprocessing',"gray2rgb");
augmentedTestSet = augmentedImageDatastore(imageResolution, testSet, ...
    'ColorPreprocessing',"gray2rgb");
%Understand Layer 2 of ResNet-50
w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);
%Convert w1(matrix) to grayscale image
% figure
% montage(w1);
% title("First Convolutional Layer Weight");
%Extract image features from layer fc1000 using activations method
%fc1000 = layer right before classification layer 
featureLayer = "fc1000";
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    "MiniBatchSize", 32, "OutputAs", "columns");
trainingLabels = trainingSet.Labels;
%Train SVM using Fit-Class Error Correcting Output Codes(ECOC).... 
%for full trained multiclass error correcting model 
classifier = fitcecoc(trainingFeatures, trainingLabels,"Learners", ...
    "linear","Coding","onevsall","ObservationsIn","columns");
%Evaluate classifier with test features
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    "MiniBatchSize", 32, "OutputAs", "columns");
%Measure accuracy of classifier 
%Pass CNN features to trained classifier  
predictLabels = predict(classifier, testFeatures, "ObservationsIn", "columns");
%Actual labels
testLabels = testSet.Labels;
%Confusion Matrix
confMat = confusionmat(testLabels, predictLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat,2));
accuracy = mean(diag(confMat));
% average accuracy obtained= 97.5%
tic
%Read new images to recognize person in pic 
cd("Testing Data")
newImage = imread(fullfile("Omar.jpg"));
cd ..
%save("net.mat", "net")
%save("classifier.mat", "classifier")


ds = augmentedImageDatastore(imageResolution, newImage, ...
    'ColorPreprocessing',"gray2rgb");
imageFeatures = activations(net, ds, featureLayer, ...
    "MiniBatchSize", 32, "OutputAs", "columns");
label = predict(classifier, imageFeatures, "ObservationsIn", "columns");
final= sprintf("The loaded image is %s", label);
display(final)
toc
