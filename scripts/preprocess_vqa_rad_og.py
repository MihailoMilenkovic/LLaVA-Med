import os
from dataclasses import dataclass
from uuid import UUID
from typing import Optional, Sequence
import random
import shutil

import json
import jsonlines


@dataclass
class VQA_RAD_dataset_entry:
    qid: str
    phrase_type: str
    qid_linked_id: UUID
    image_case_url: str
    image_name: str
    image_organ: str
    evaluation: str
    question: str
    question_type: str
    answer: str
    answer_type: str
    question_rephrase: Optional[str] = ""
    question_relation: Optional[str] = ""
    question_frame: Optional[str] = ""

    def __post_init__(self):
        self.answer = str(self.answer).strip().lower()
        self.question = str(self.question).strip().lower()


@dataclass
class VQA_RAD_dataset:
    data: Sequence[VQA_RAD_dataset_entry]
    og_image_location: Optional[str] = ""

    def filter_data(self):
        # NOTE: Currently only sticking to yes/no questions
        # should consider adding other info later
        self.data = [d for d in self.data if d.answer in ["yes", "no"]]

    def __post_init__(self):
        self.filter_data()

    @classmethod
    def from_json(cls, filename: str, og_image_location: str):
        with open(filename, "r") as f:
            data = json.load(f)
            entries = [VQA_RAD_dataset_entry(**entry) for entry in data]
            return cls(data=entries, og_image_location=og_image_location)

    def to_conversation_json(
        self, save_path: os.PathLike, include_question_type: Optional[bool] = True
    ):
        json_data = []
        for i, data in enumerate(self.data):
            curr_json_data = {
                "id": i,
                "image": data.image_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{data.question}\n<image>",
                    },
                    {"from": "gpt", "value": data.answer},
                ],
            }
            if include_question_type:
                curr_json_data["question_type"] = data.question_type
            json_data.append(curr_json_data)
        # Save JSON data
        with open(os.path.join(save_path), "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        print("saved data to ", json_file)

    def train_test_split(self, test_ratio: Optional[float] = 0.2):
        # NOTE: using different images for training and testing

        # random.shuffle(self.data)
        self.data = sorted(self.data, key=lambda x: x.image_name)
        split_index = int(len(self.data) * (1 - test_ratio))
        while (
            self.data[split_index - 1].image_name == self.data[split_index].image_name
        ):
            split_index += 1

        train_data = self.data[:split_index]
        test_data = self.data[split_index:]
        return VQA_RAD_dataset(
            train_data, og_image_location=self.og_image_location
        ), VQA_RAD_dataset(test_data, og_image_location=self.og_image_location)

    def to_jsonl(self, filename: str, include_question_type: Optional[bool] = True):
        with jsonlines.open(filename, "w") as f:
            for i, entry in enumerate(self.data):
                new_obj = {
                    "question": entry.question,
                    "image": entry.image_name,
                    "question_id": i,
                }
                if include_question_type:
                    new_obj["question_type"] = entry.question_type
                f.write(new_obj)
        print(f"saved data to {filename}")

    def save_images(self, image_folder: str):
        all_images = [d.image_name for d in self.data]

        os.makedirs(image_folder, exist_ok=True)
        for image_name in all_images:
            shutil.copyfile(
                os.path.join(self.og_image_location, image_name),
                os.path.join(image_folder, image_name),
            )

    def save(self, save_folder: str, include_question_type: Optional[bool] = True):
        self.to_jsonl(
            os.path.join(save_folder, "questions.jsonl"),
            include_question_type=include_question_type,
        )
        self.to_conversation_json(
            os.path.join(save_folder, "data.json"),
            include_question_type=include_question_type,
        )
        self.save_images(os.path.join(save_folder, "images"))


if __name__ == "__main__":

    dataset_folder_location = "/home/mmilenkovic/git/LLaVA-Med/data/vqa_rad_data"
    dataset_json_location = os.path.join(
        dataset_folder_location, "VQA_RAD Dataset Public.json"
    )
    dataset_save_location = os.path.join(
        os.path.dirname(dataset_folder_location), "vqa_rad"
    )
    os.makedirs(dataset_save_location, exist_ok=True)

    dataset_images_location = os.path.join(dataset_folder_location, "images")
    dataset_train_location = os.path.join(dataset_save_location, "train")
    dataset_test_location = os.path.join(dataset_save_location, "test")
    os.makedirs(dataset_train_location, exist_ok=True)
    os.makedirs(dataset_test_location, exist_ok=True)

    dataset = VQA_RAD_dataset.from_json(
        dataset_json_location, og_image_location=dataset_images_location
    )

    train_dataset, test_dataset = dataset.train_test_split()
    test_images = set(d.image_name for d in test_dataset.data)
    train_images = set(d.image_name for d in train_dataset.data)
    print("intersection:", test_images.intersection(train_images))

    train_dataset.save(dataset_train_location, include_question_type=True)
    test_dataset.save(dataset_test_location, include_question_type=True)
