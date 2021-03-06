dir_ts      = "/home/hyou/localrepo/glasso/fs125/992774/timeseries/tfMRI_LANGUAGE_125mm_LR/Lausanne2008/ROIv_scale33";
output_file = "test_mean.ts.matlab";

% load the first timeseries file to extract dimension information
ts_raw      = importdata(dir_ts + "/1.txt");
ts_list     = dir(dir_ts + "/*.txt");
num_frame   = size(ts_raw, 2) - 6; %remove coordinate prefix
num_roi     = size(ts_list, 1);

ts_mean = zeros(num_roi, num_frame);
for i=1:num_roi
    ts_raw       = importdata(dir_ts+"/"+num2str(i)+".txt");
    nzn_voxel    = find(sum(ts_raw(:,7:end), 2) ~= 0);
    ts_mean(i,:) = mean(ts_raw(nzn_voxel, 7:end)); 
end

dlmwrite(dir_ts+'/'+output_file, ts_mean, 'precision','%0.3f');