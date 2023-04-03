close all
clear all;

stim_dir = '/home/mayajas/Documents/project-0b-pRF-tutorial-3T/code/analysis-scripts/matlab/';
mkdir([stim_dir 'bar_run-01'])
mkdir([stim_dir 'bar_run-02'])
mkdir([stim_dir 'bar_run-03'])


% load([behav_dir 'data20191021T105933.mat'])
% load([behav_dir 'data20191021T112726.mat'])
load([stim_dir 'params_1500_02.mat'])
orig_size = size(original_stimulus.images{1});
img = original_stimulus.images{1};
stim_size = 200;
% stim = zeros(orig_size(1),orig_size(2),...
%     size(stimulus.seq,1)/60);
stim = zeros(stim_size,stim_size,...
    size(stimulus.seq,1)/30);
size(stim)
%%
figure
i = 1;
for f = 1:30:size(stimulus.seq,1)
  c = squeeze(img(:,:,stimulus.seq(f)));
  idx = zeros(size(c));
  idx(c==1 | c==254) = 1;
  stim(:,:,i) = imresize(idx, [200 200], 'nearest');
%   imagesc(stim(:,:,i)); colorbar;
%   pause(1);
  %f
  i = i + 1;
end

% stim = stim(:,:,1:60:size(stim,3));
stim = stim(:,:,9:end);
% figure
% for f = 1:size(stim,3)
%   imagesc(stim(:,:,f)); colorbar;
%   pause(0.5);
%   imwrite(stim(:,:,f),[stim_dir 'bar_run-01' filesep 'frame_00' num2str(f) '.png'])
%   imwrite(stim(:,:,f),[stim_dir 'bar_run-02' filesep 'frame_00' num2str(f) '.png'])
%   imwrite(stim(:,:,f),[stim_dir 'bar_run-03' filesep 'frame_00' num2str(f) '.png'])
% end

%save('stimulus_bar','stim');
ApFrm=stim;
save('aps_samsrf.mat','ApFrm');