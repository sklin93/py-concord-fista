
% check union of subjects list across all tasks (and streamline data)

subject_list_uni = []
subject_list_int = []

dataset_dir = '~/gdrive/glasso/data/HCP-V1/'
dataset_list = dir([dataset_dir, '*.mat'])
for i = 1:length(dataset_list)
    load([dataset_dir, dataset_list(i).name]);
    if (i == 1)
        subject_list_uni = subj;
        subject_list_int = subj;
    end
    subject_list_uni = union(subject_list_uni, subj);
    subject_list_int = intersect(subject_list_int, subj);
end

disp('union');
disp(subject_list_uni);
disp('intersection');
disp(subject_list_int);
