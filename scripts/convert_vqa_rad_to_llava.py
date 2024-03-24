import os
import argparse
import json
from PIL import Image
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_dir", type=str, required=True, help="Directory to save train data"
    )
    ap.add_argument(
        "--test_dir", type=str, required=True, help="Directory to save test data"
    )
    args = ap.parse_args()
    # Load the dataset
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    # Create folders for train and test splits
    train_folder = args.train_dir
    train_image_folder = os.path.join(args.train_dir, "images")
    test_folder = args.test_dir
    test_image_folder = os.path.join(args.test_dir, "images")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(test_image_folder, exist_ok=True)
    json_data = []
    # Process train and test splits
    for split in ["train", "test"]:
        for i, data in enumerate(dataset[split]):
            image_id = str(i)
            image_path = f"{image_id}.jpg"  # Path to save the image
            image_data = data["image"]
            image_data.save(
                os.path.join(
                    train_image_folder if split == "train" else test_image_folder,
                    image_path,
                )
            )  # Save image

            # Create JSON data
            curr_json_data = {
                "id": image_id,
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{data['question']}"},
                    {"from": "gpt", "value": data["answer"]},
                ],
            }
            json_data.append(curr_json_data)
        # Save JSON data
        with open(
            os.path.join(
                train_folder if split == "train" else test_folder,
                f"{split}.json",
            ),
            "w",
        ) as json_file:
            json.dump(json_data, json_file, indent=4)


if __name__ == "__main__":
    main()
