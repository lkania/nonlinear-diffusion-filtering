img = imread("image_mask.png");
img_g = rgb2gray(img);
img_d = im2double(img_g);

% mask, 0 is not red, 1 is red
mask = img(:,:,1)==252 & img(:,:,2)==2 & img(:,:,3)==4;

iterations = 500;
dt = 0.19;
eps = 1;

I = run(img_d.*(~mask),mask,dt,eps,iterations);

f = figure('Position',[0, 0, 700, 300]);
subplot(1,2,1)
imshow(img_d)
title('Original Image')
subplot(1,2,2)
imshow(I)
title('After Total Variation Inpainting')
saveas(f,'sol.png');

function [C_]=rowflatten(C)
 C_ = reshape(C.',1,[]);
end

function I=run(img,mask,dt,eps,iterations)
    I = repmat(img,1);
    
    [N,M] = size(I);
    
    eps2 = eps^2;
    
    for it = 1:1:iterations
        
        fprintf('Iteration %i\n', it);
        
        Iss = circshift(I,1,1); %   matrix shifted to the south
        Iss(1,:) = zeros(1,M);
        
        Isn = circshift(I,-1,1);%   matrix shifted to the north
        Isn(N,:) = zeros(1,M);
        
        Isw = circshift(I,-1,2);%   matrix shifted to the west
        Isw(:,M) = zeros(1,N).';
        
        Ise = circshift(I,1,2);%    matrix shifted to the east
        Ise(:,1) = zeros(1,N).';
        
        In = Iss-Isn; 
        Ie = Isw-Ise; 
        
        In2 = In.^2;
        Ie2 = Ie.^2;
        
        c1 = eps2 + 0.25 * In2;
        c2 = eps2 + 0.25 * Ie2;
        c3 = - 0.125 .* Ie .* In;
        
        r = eps2 + 0.25 .* (Ie2 + In2);
        rinv = 1 ./ r.^1.5;
        rinvdt = dt .* rinv; 
        
        cd = 1 - rinvdt .* 2 .* (c1+c2);
        
        ca = - rinvdt .* c3;
        cp = ca;
        
        ca(:,1) = zeros(1,N).'; 
        ca(1,:) = zeros(1,M);
        
        cp(:,M) = zeros(1,N).';
        cp(N,:) = zeros(1,M);
        
        cb = rinvdt .* c2;
        cg = cb;
        
        cb(1,:) = zeros(1,M);    
        cg(N,:) = zeros(1,M);
        
        cc = rinvdt .* c1;
        ce = cc;
        
        cc(:,1) = zeros(1,N).';
        ce(:,M) = zeros(1,N).';
        
        cq = rinvdt .* c3;
        ch = cq;
        
        cq(:,M) = zeros(1,N).';
        cq(1,:) = zeros(1,M);    
        
        ch(:,1) = zeros(1,N).';
        ch(N,:) = zeros(1,M);
        
        % row-flatten all coefficients and the image
        
        ca = rowflatten(ca).';
        cb = rowflatten(cb).';
        cq = rowflatten(cq).';
        cc = rowflatten(cc).';
        cd = rowflatten(cd).';
        ce = rowflatten(ce).';
        ch = rowflatten(ch).';
        cg = rowflatten(cg).';
        cp = rowflatten(cp).';
        
        I = rowflatten(I).';
        
        % create diagonal matrix for jacobi iteration
        
        columns = horzcat(ca,cb,cq,cc,cd,ce,ch,cg,cp);
        
        Q = spdiags(columns,[-(M+2) -(M+1) -M -1 0 1 M M+1 M+2],N*M,N*M);
        
        I = Q * I;
        
        I = reshape(I.',M,N).';
        I = img.*(~mask)+I.*(mask);
       
    end

end


