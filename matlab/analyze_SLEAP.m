%% Set up directories
clear;

h5_dir = "/Users/ellasay/Google Drive/My Drive/1.SLEAP/h5";
cd(h5_dir)

video_dir = "/Users/ellasay/Google Drive/My Drive/1.SLEAP/Videos";
cd(video_dir)

h5_filenames = dir(fullfile(h5_dir, '*.h5'));
h5_filenames = {h5_filenames.name}; h5_filenames = h5_filenames';

video_filenames_struct = dir(fullfile(video_dir,'*.mp4'));
video_filenames = fullfile({video_filenames_struct.folder}', {video_filenames_struct.name}');


%% Import SLEAP h5 file

h5_files = import_h5(h5_dir);
% pos = h5_files.Content{1,1}.tracks(val);

% h5_files.Tracks = cell(height(h5_files),1);
% h5_files.VideoPath = cell(height(h5_files),1);

for i = 1:height(h5_files)
    % add tracks to dataframe
    structVar = h5_files.Content{i};
    h5_files.Tracks{i} = structVar.tracks;

    % add video path (matching names)
    filename = extractBetween(h5_files.FileName(i),"",".mp4"); 
    matchIdx = find(contains(video_filenames, filename));  % Find index of the match
    if ~isempty(matchIdx)
        h5_files.VideoPath{i} = video_filenames{matchIdx(1)}; 
    else
        h5_files.VideoPath{i} = 'No match found';  
    end

end
%% Import SLEAP (not used)

cd(h5_dir);
for i=1:size(h5_filenames)
    filename = h5_filenames{i};
    
    % Open the HDF5 file in read-only mode
    info = h5info(filename);
    
    % Get the list of dataset names
    dset_names = {info.Datasets.Name};
    
    locations = h5read(filename, '/tracks'); 
    
    % Read the 'node_names' dataset and decode it (if it's stored as strings)
    node_names_raw = h5read(filename, '/node_names');
    node_names = cellfun(@char, cellstr(node_names_raw), 'UniformOutput', false); % Convert to cell array of strings
   
end


%% track one mouse

%% 