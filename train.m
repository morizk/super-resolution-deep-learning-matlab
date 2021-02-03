imm=imread('data4 (2).bmp');
x=rgb2gray(imm);
imm=[];
[a, aa]=size(x);
%%
zoom=1.75;

im=imresize(x,1/zoom);

[hi ,le]=size(im);

newhi=round(zoom*hi);
newle=round(zoom*le);
imm=zeros(newhi,newle);
reh=hi/newhi;
rel=le/newle;

for h=1:newhi
for l=1:newle
oh= round(reh*h);
if oh==0
oh=1;
end
ol=round(rel*l);
if ol==0
ol=1;
end
imm(h,l)=im(oh,ol);
end
end

imm=imresize(imm,[a aa]);

imshow([imm,x])
%%
segment_size=10;
double_segment=segment_size*2;

d=size(imm);
r=zeros(d(1),segment_size);
imm=[r,imm,r];
d=size(imm);
r=zeros(segment_size,d(2));
imm=[r;imm;r];

%%
n=1;

in=[];
out=[];
for i=1:d(1)
for j=1:d(2)-double_segment
in(:,:,n)=imm(i:i+double_segment,j:j+double_segment);
out(n,1)=x(i,j);
n=n+1;
end
end
sizeOfData=size(in,3);
in=reshape(in,[double_segment+1,double_segment+1,1,sizeOfData]);
out=categorical(out);

%%

layers = [
    imageInputLayer([double_segment+1 double_segment+1],'Name','input')
    
    convolution2dLayer(3,512,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxPooling1')

     convolution2dLayer(3,1024,'Padding','same','Name','conv_3')
     batchNormalizationLayer('Name','BN_3')
     reluLayer('Name','relu_3')
    
    convolution2dLayer(3,2048,'Padding','same','Name','conv_4')
    batchNormalizationLayer('Name','BN_4')
    reluLayer('Name','relu_4')  
    
    depthConcatenationLayer(2,'Name','concat');
    fullyConnectedLayer(256,'Name','conntct')
    softmaxLayer('Name','softmaxLayer')
    classificationLayer('Name','classificationLayer')
    ];

lgraph = layerGraph(layers);
 color = crop2dLayer('centercrop','Name','color');
 lgraph = addLayers(lgraph,color);
 lgraph = connectLayers(lgraph,'input','color/in'); 
 lgraph = connectLayers(lgraph,'relu_4','color/ref');  
 lgraph = connectLayers(lgraph,'color','concat/in2');
%%

idx = randperm(sizeOfData,round(sizeOfData/3));
XValidation = in(:,:,:,idx);
in(:,:,:,idx) = [];  
YValidation = out(idx);
out(idx) = [];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0009, ...
    'MaxEpochs',400, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',5000, ...
    'Verbose',false, ...
    'Plots','training-progress');
figure
plot(lgraph)

%%

net = trainNetwork(in,out,lgraph,options);

