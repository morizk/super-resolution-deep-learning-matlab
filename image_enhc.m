x=imread('TEST_3.PNG');
%%

zoom=1.7;

zoomed_img=imresize(x,zoom,'nearest');


compere=zoomed_img;
final_size=size(zoomed_img);

%%
segment_size=7;
double_segment=segment_size*2;
d=size(zoomed_img);
r=zeros(d(1),segment_size,3);
zoomed_img=[r,zoomed_img,r];
d=size(zoomed_img);
r=zeros(segment_size,d(2),3);
zoomed_img=[r;zoomed_img;r];

%%
n=1;

in_r=[];
in_b=[];
in_g=[];

for i=1:d(1)
for j=1:d(2)-double_segment
in_r(:,:,n)=zoomed_img(i:i+double_segment,j:j+double_segment,1);
in_b(:,:,n)=zoomed_img(i:i+double_segment,j:j+double_segment,2);
in_g(:,:,n)=zoomed_img(i:i+double_segment,j:j+double_segment,3);

n=n+1;

end
end
sizeOfData=size(in_r,3);
in_r=reshape(in_r,[double_segment+1,double_segment+1,1,sizeOfData]);
in_b=reshape(in_b,[double_segment+1,double_segment+1,1,sizeOfData]);
in_g=reshape(in_g,[double_segment+1,double_segment+1,1,sizeOfData]);


    
%%

out_r = classify(net,in_r);
out_r=grp2idx(out_r);
out_b = classify(net,in_b);
out_b=grp2idx(out_b);
out_g = classify(net,in_g);
out_g=grp2idx(out_g);
final_image(:,:,1)=uint8(flip(rot90(reshape(out_r,[final_size(2),final_size(1)]))));
final_image(:,:,2)=uint8(flip(rot90(reshape(out_b,[final_size(2),final_size(1)]))));
final_image(:,:,3)=uint8(flip(rot90(reshape(out_g,[final_size(2),final_size(1)]))));

imshow([uint8(final_image),uint8(compere)])
