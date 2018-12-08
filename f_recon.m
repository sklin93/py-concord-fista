% load data
task = 'EMOTION';
[vec_s, vec_f] = load_data(task);

% load omega
fdir = 'fs_results/';
addpath('/home/sikun/Downloads/npy-matlab/npy-matlab')
omega = readNPY(strcat(fdir,'0.0014_1stage_er2_',task,'.npy'));
[p,~] = size(omega);
omega = omega(:,p+1:end);
size(omega)
nnz(omega)

% using cvx
addpath('/home/sikun/Downloads/cvx')
cvx_setup
lam = 0.01;
lamMat = ones(p,p)*lam;
lamMat(omega==0) = lam*10000;
cvx_begin
    variable X(p,p)
    minimize(sum_square(vec_s*X'-vec_f)+sum(lamMat.*(X.^2)));
cvx_end
size(X)
function [vec_s, vec_f] = load_data(task)
    if task=='resting'
        data_dir = '/home/sikun/Documents/data/Bassette/';
        data = load(strcat(data_dir,'data_matrices.mat'));
        sMat = data.Ss;
        s = cat(3,sMat{:});
		fMat = data.Fs;
        f = cat(3,fMat{:});
    else
        data_dir = '/home/sikun/Documents/data/HCP-V1/';	
        sMat = load(strcat(data_dir,'ER_S_direct.mat'));
        s = sMat.X;
		fMat = load(strcat(data_dir,'tfMRI-',task,'.mat'));
        f = fMat.X;
    end
    [d,~,n] = size(s);
    p = d*(d-1)/2;
    vec_s = zeros(n,p);
    vec_f = zeros(n,p);
    for subj = 1:n
        ctr = 1;
        for i = 1:d-1
            for j = i+1:d
              vec_s(subj,ctr) = s(i,j,subj);
              vec_f(subj,ctr) = f(i,j,subj);
              ctr = ctr + 1;
            end
        end
    end
end