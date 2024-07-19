import os
import json

def combine_json_files(directory, output_file):
    combined_data = {}

    # Iterate through all files in the directory
    for i in range(1,360):
        filename = f"transforms_movie{i}.jsons"
        file_path = os.path.join(directory, f"transforms_movie{i}.json")
        with open(file_path, 'r') as file:
            data = json.load(file)
            combined_data[filename] = data

    # Write the combined data into a single output JSON file
    with open(os.path.join(directory, output_file), 'w') as output_file:
        json.dump(combined_data, output_file, indent=4)

# Directory containing the JSON files

scenes = ["food_raxel", "cinema_raxel", "chef_raxel", "carving_raxel", "boots_raxel", "baskets_raxel"]
for scene in scenes:
    directory = f"/scratch/ondemand28/weihanluo/transientangelo/load/captured_data/{scene}/final_cams/movie_jsons/"

    # Output file name
    output_file = "transforms_movie_combined.json"

    # Combine the JSON files
    combine_json_files(directory, output_file)
