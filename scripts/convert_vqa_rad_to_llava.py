import os
import argparse
import json
from typing import Optional
from PIL import Image
from datasets import load_dataset, Dataset


def save_data_to_dir(dataset: Dataset, output_dir: str):
    json_data = []
    for i, data in enumerate(dataset):
        image_id = str(i)
        image_path = f"{image_id}.jpg"  # Path to save the image
        image_data = data["image"]
        image_data.save(
            os.path.join(
                output_dir,
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
            output_dir,
            "data.json",
        ),
        "w",
    ) as json_file:
        json.dump(json_data, json_file, indent=4)
    return json_data


def prepare_test_questions(
    dataset: Dataset, output_folder: str, output_name: Optional[str] = "questions.jsonl"
):
    output_data = []
    for i, data in enumerate(dataset):
        image_id = str(i)
        text = data["question"]
        category = "medical-vqa"
        new_data = {"question_id": image_id, "text": text, "category": category}
        output_data.append(new_data)
    output_path = os.path.join(output_folder, output_name)
    with open(output_path, "w") as file:
        for item in output_data:
            json.dump(item, file)  # Write item as JSON
            file.write("\n")


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
    train_data = dataset["train"]
    test_data = dataset["test"]

    save_data_to_dir(train_data, train_folder)
    save_data_to_dir(test_data, test_folder)
    prepare_test_questions(test_data, output_folder=test_folder)


if __name__ == "__main__":
    main()
