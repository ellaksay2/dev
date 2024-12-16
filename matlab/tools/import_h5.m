function dataFrame = import_h5(directoryPath)
    % Get list of all h5 files in the directory
    h5Files = dir(fullfile(directoryPath, '*.h5'));
    
    % Initialize a table with two columns: FileName and Content
    dataFrame = table('Size', [length(h5Files) 2], 'VariableTypes', {'string', 'cell'}, ...
                      'VariableNames', {'FileName', 'Content'});
    
    % Loop over each file
    for i = 1:length(h5Files)
        fileName = h5Files(i).name;
        filePath = fullfile(directoryPath, fileName);
        
        % Store file name in the FileName column
        dataFrame.FileName(i) = fileName;
        
        % Get list of datasets in the h5 file
        info = h5info(filePath);
        
        % Initialize a structure to store all datasets in this file
        fileContent = struct();
        
        % Loop through datasets and store the data
        for j = 1:length(info.Datasets)
            datasetName = info.Datasets(j).Name;
            datasetPath = ['/' datasetName]; % Define dataset path within h5 file
            
            % Read the dataset
            data = h5read(filePath, datasetPath);
            
            % Store the data in the fileContent structure
            fileContent.(datasetName) = data;
        end
        
        % Store the file content in the Content column
        dataFrame.Content{i} = fileContent;

        
    end
end
