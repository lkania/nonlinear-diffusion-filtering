img = imread("mri.png");
img_g = rgb2gray(img);
img_d = im2double(img_g);

f = figure('Position',[0, 0, 700, 900],'DefaultAxesPosition', [0.1, 0.1, 0.8, 0.8]);

subplot(5,2,1)
imshow(img_d)
str = sprintf('Original Image');
title(str);

Ks = [0.01 0.01 0.1 0.1];
its = [200,10,200,200];
dts = [0.25,5,5,5];

for i = 1:1:4 
    K = Ks(i);
    dt = dts(i);
    it = its(i);
    res = enld(img_d,K,dt,it);
    subplot(5,2,2*i+1)
    imshow(res)
    str = sprintf('Explicit K=%0.2f Its=%i dt=%0.2f', K,it,dt);
    title(str);

    subplot(5,2,2*i+2)
    res = sinld(img_d,K,dt,it);
    imshow(res)
    str = sprintf('Semi-implicit K=%0.2f Its=%i dt=%0.2f', K,it,dt);
    title(str);
end

saveas(f,'sol.png');

function [Iss,Isn,Isw,Ise]=shifts(I)
    Iss = circshift(I,1,1); %   matrix shifted to the south
    Isn = circshift(I,-1,1);%   matrix shifted to the north
    Isw = circshift(I,-1,2);%   matrix shifted to the west
    Ise = circshift(I,1,2);%    matrix shifted to the east
end

function [In,Is,Ie,Iw]=grads(I)
    [Iss,Isn,Isw,Ise] = shifts(I);

    In = Iss-I; % gradient in the north direction
    Is = Isn-I; % gradient in the south direction
    Ie = Isw-I; % gradient in the east direction
    Iw = Ise-I; % gradient in the west direction
end

function [Cn,Cs,Ce,Cw]=coeffs(I)
    [Css,Csn,Csw,Cse] = shifts(I);

    Cn = (Css+I)./2; 
    Cs = (Csn+I)./2; 
    Ce = (Csw+I)./2; 
    Cw = (Cse+I)./2; 
end

% Explicit implementation
function I=enld(img,K,dt,iterations)
    I = repmat(img,1);
   
    for it = 1:1:iterations 
        
        % using central difference for computing the magnitude in C
        Iss = circshift(I,1,1);  %   matrix shifted to the south
        Isn = circshift(I,-1,1); %   matrix shifted to the north
        Isw = circshift(I,-1,2); %   matrix shifted to the west
        Ise = circshift(I,1,2);  %   matrix shifted to the east
        
        In = Iss-I; % gradient in the north direction
        Is = Isn-I; % gradient in the south direction
        Ie = Isw-I; % gradient in the east direction
        Iw = Ise-I; % gradient in the west direction
        
        Inc = (Iss - Isn) ./ 2;
        Iec = (Isw - Ise) ./ 2;
 
        C = 1 ./ (1 + (Inc.^2 + Iec.^2) / K^2);
       
        % Alternatively, forward differences can be used for computing 
        % for computing the magnitude in C
        % [In,Is,Ie,Iw] = grads(I);
        % C = 1 ./ (1 + (In.^2 + Ie.^2) / K^2);
        
        [Cn,Cs,Ce,Cw] = coeffs(C);
        
        I = I + dt * (In .* Cn + Is .* Cs + Ie .* Ce + Iw .* Cw);
    end
end

function [upper,diagonal,lower] = diagonals(C,dt)

    Csw = circshift(C,-1,2);
    Cse = circshift(C,1,2);

    upper = - dt * 2 * (C + Csw);
    lower = - dt * 2 * (Cse + C);
    diagonal = 2 + dt * ( 4 * C + 2 * Csw + 2 * Cse);
end

% Semi-implicit implementation
function I=sinld(img,K,dt,iterations)
    I = repmat(img,1);
    
    [N,M] = size(I);
   
    for it = 1:1:iterations 
        
        Iss = circshift(I,1,1);
        Isw = circshift(I,-1,2);
        
        In = Iss-I;
        Ie = Isw-I;
        
        C = 1 ./ (1 + (In.^2 + Ie.^2) ./ K^2);
                
        tmp = 0;
        
        % flatten C row-wise
        
        Crow = reshape(C.',1,[]);
        
        [upper,diagonal,lower] = diagonals(Crow,dt);
        
        sol = tridiag(diagonal,lower,upper,reshape(I.',1,[]));
        
        tmp = tmp + reshape(sol,M,N)';
        
        % flatten C column-wise
        
        Ccol = reshape(C,1,[]);
        
        [upper,diagonal,lower] = diagonals(Ccol,dt);
        
        sol = tridiag(diagonal,lower,upper,reshape(I,1,[]));
        
        tmp = tmp + reshape(sol,N,M);
        
        I = tmp; 
      
    end
end



