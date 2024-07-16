import os

base_dir = "vector_jobs_captured_10"

scenes = ["baskets_raxel", "boots_raxel", "carving_raxel", "chef_raxel", "cinema_raxel", "food_raxel"]
views = ["two_views", "three_views", "five_views"]

for scene in scenes:
    for view in views:
        path = os.path.join(base_dir, scene, view)
        os.makedirs(path, exist_ok=True)
print("Directory initialized")


