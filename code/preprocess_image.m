function imgOut = preprocess_image(img, imgSize)
% Resize, denoise, and normalize image
if size(img,3)==1, img = repmat(img, [1 1 3]); end
img = im2double(imresize(img, imgSize));
for ch=1:3, img(:,:,ch) = medfilt2(img(:,:,ch), [3 3]); end
h = fspecial('gaussian',[5 5],1.0);
blur = imfilter(img, h, 'replicate');
img = img + 0.3*(img - blur);
img = max(0,min(1,img));
imgOut = img;
end
