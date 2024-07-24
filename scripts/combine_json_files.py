import os
import json

def combine_json_files(directory, output_file, sim):
    '''
    Combine the json files for movie for each dataset into a single json file.
    If the dataset is sim, then i ranges from 0-359, else i ranges from 1-360.
    '''
    if sim:
        selected_range = range(0,359)
    else:
        selected_range = range(1,360)
    combined_data = {}

    # Iterate through all files in the directory
    for i in selected_range:
        filename = f"transforms_movie{i}.jsons"
        file_path = os.path.join(directory, f"transforms_movie{i}.json")
        with open(file_path, 'r') as file:
            data = json.load(file)
            combined_data[filename] = data

    # Write the combined data into a single output JSON file
    with open(os.path.join(directory, output_file), 'w') as output_file:
        json.dump(combined_data, output_file, indent=4)

# Directory containing the JSON files

scenes_cap = ["food_raxel", "cinema_raxel", "chef_raxel", "carving_raxel", "boots_raxel", "baskets_raxel"]
scenes_sim = ["lego", "benches", "chair", "ficus", "hotdog"]


if __name__ == "__main__":
    for scene in scenes_sim:
        directory = f"/scratch/ondemand28/weihanluo/transientangelo/load/transient_nerf_synthetic/{scene}/{scene}_jsons/movie_jsons/"

        # Output file name
        output_file = "transforms_movie_combined.json"

        # Combine the JSON files
        combine_json_files(directory, output_file, sim=True)
