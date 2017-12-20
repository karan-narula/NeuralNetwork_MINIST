%The purpose of this function is to read the training set images and
%training set labels from the binary file and output a training vector,
%validation vector and their corresponding labels in a usuable format to
%the neural network training and validation.

%the training files and test files must be in the same folder as this
%function

%The outputs are :
%X -> is a 2 dimensional array, total_pixels x data_sets dimension. The values in X
     %has been scaled from [0 255] to [0 1] for the purpose of NN training
%Y -> is 2 dimensionaly array, 10 x data_sets dimension which contains in each row
    %an output of one on the neuron corresponding to the digit (0-9)
%images -> contains the images read from the binart file. The total number
        %of images are 60000 and the structure of this array is 28x28x60000 for
        %each 28x28 images. The way to display the image has been shown in
        %the debug section
%For the inputs, there is flag for debugging. The debuggin allows for
%showing the images being correspond to the labels correctly. It is an
%additional parameter and is not necessary to be specified in this function

function [X,Y, images] = ReadData(scale,varargin)
if(nargin > 1)
    flag_debug = varargin{1};
else
    flag_debug = 0;
end

%fix the filenames that will be read
TrainingSet_img_filename = 'train-images.idx3-ubyte';
TrainingSet_label_filename = 'train-labels.idx1-ubyte';
TestSet_label_filename = 't10k-labels.idx1-ubyte';
TestSet_img_filename = 't10k-images.idx3-ubyte';

%firstly read the training images file
fileID = fopen(TrainingSet_img_filename, 'r' ,'ieee-be');
%first 32 bits (int32) is a magic number which is of no consequence
temp = fread(fileID, 1, 'int32');
if(temp ~= 2051)
    error('Training File Format is incorrect \n');
end    
%second, 32 bit integer specifying the number of images in the file
num_item = fread(fileID, 1, 'int32');
%third, 32 bit ingeger specifying the number of rows
num_rows = fread(fileID, 1, 'int32');
%fourth, 32 bit integer specifying the number of columns
num_columns = fread(fileID, 1, 'int32');
X = zeros(num_rows*num_columns*scale^2,num_item);
images = zeros(num_rows*scale,num_columns*scale,num_item);
%iteratively read an image at a time
for i =1:num_item
   images(:,:,i) = imresize(fread(fileID, [num_rows,num_columns], 'uint8')', scale);
   X(:,i) = double(reshape(images(:,:,i)',[1,num_rows*num_columns*scale^2]));
end
%normalise X to be between 0 and 1
X = X/255;
%close the file
fclose(fileID);

%read the label of the training images
fileID = fopen(TrainingSet_label_filename, 'r' ,'ieee-be');
%first 32 bits (int32) is a magic number which is of no consequence
temp = fread(fileID, 1, 'int32');
if(temp ~= 2049)
    error('Label File Format is incorrect \n');
end
%second, 32 bit integer specifying the number of items (which would be the
%same as training data set)
num_item = fread(fileID, 1, 'int32');
Y = double(fread(fileID, [1,num_item], 'uint8'));
fclose(fileID);

%if someone wants to debug that the images are being Read correct then
%display random images with their labels on the tittles
if(flag_debug)
   figure(1); clf; hold on;
   num_random = 6;
   image_no = randi([1,num_item],[num_random,1]);
%    image_no = 1:num_random; %first few images are easier to see by eyes
   for i =1:num_random
       subplot(3,2,i); hold on;
       imagesc(flipud(images(:,:,image_no(i))));
       colormap gray;
       title(['The label for this image is ',num2str(Y(image_no(i)))]);
       set(gca, 'Xlim',[1,num_columns*scale]);
       set(gca, 'Ylim', [1,num_rows*scale]);
       axis off;
   end
end

%adjust the Y to be in the format required by Neural Network 10xnum_data
indx = (0:num_item-1)*10;
Y_adjusted = zeros(10,num_item);
Y_adjusted(indx+Y+1) = 1;
Y = Y_adjusted;
