function [ptitle, pdescrip, RMSE] = my_fitnlm(path, b1, b2)
load(path)
X = [st_r', sd_r', st', sd'];
y = single(ratings);
modelfun = @(b,x)(b(1)*X(:,1)+b(2)*X(:,2)) ./ (b(1)*X(:,3) + b(2)*X(:,4));
beta0 = [b1,b2];
mdl = fitnlm(X, y, modelfun, beta0);
ptitle = mdl.Coefficients.Estimate(1);
pdescrip = mdl.Coefficients.Estimate(2);
RMSE = mdl.RMSE;