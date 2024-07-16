import os
import subprocess

# Source and target directories
source_dir = 'luowei11@v.vectorinstitute.ai:/projects/transientangelo/rsync_process.py'
target_dir = '/scratch/ondemand28/weihanluo/transientangelo/monosdf_sim_without_mask'

# Rsync options
rsync_options = '-avz'

# Specify the range
start_folder = '12792630'
end_folder = '12798176'

def get_folders_between(start_folder, end_folder, source_dir):
    
    #TODO: Need to ssh first into the server
    
    all_folders = sorted(os.listdir(source_dir))
    selected_folders = []
    start_index = None
    end_index = None

    if start_folder in all_folders:
        start_index = all_folders.index(start_folder)
    if end_folder in all_folders:
        end_index = all_folders.index(end_folder)

    if start_index is not None and end_index is not None and start_index < end_index:
        selected_folders = all_folders[start_index + 1:end_index]
    
    return selected_folders

def rsync_folders(folder_names, source_dir, target_dir):
    for folder in folder_names:
        source_path = os.path.join(source_dir, folder)
        target_path = os.path.join(target_dir, folder)
        command = f'rsync {rsync_options} {source_path} {target_path}'
        
        print(f'Executing: {command}')
        subprocess.run(command, shell=True, check=True)

if __name__ == '__main__':
    folder_names = get_folders_between(start_folder, end_folder, source_dir)
    print(folder_names)
    # rsync_folders(folder_names, source_dir, target_dir)
