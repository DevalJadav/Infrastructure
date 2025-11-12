function model = train_knn(X,y,k)
if nargin<3, k=5; end
model.X = X; model.y = y; model.k = k;
end
