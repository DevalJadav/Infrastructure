function edges = edge_detect(gray)
Gx = [-1 0 1; -2 0 2; -1 0 1];
Gy = Gx';
Ix = conv2(double(gray), Gx, 'same');
Iy = conv2(double(gray), Gy, 'same');
mag = sqrt(Ix.^2 + Iy.^2);
th = mean(mag(:)) * 1.2;
edges = mag > th;
end
