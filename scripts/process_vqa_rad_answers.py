import json
import argparse
from typing import Sequence, List, Optional
from dataclasses import dataclass, field

import jsonlines

import matplotlib.pyplot as plt


@dataclass
class AnswerComparison:
    question: str
    llm_answer: str
    real_answer: str
    question_type: str
    question_types: Optional[List[str]] = None

    def replace_with_yes_or_no(self, s: str):
        yes_index = s.find("yes")
        no_index = s.find("no")

        if yes_index != -1 and (no_index == -1 or yes_index < no_index):
            return "yes"
        elif no_index != -1:
            return "no"
        else:
            return "no"

    def __post_init__(self):
        self.llm_answer = self.llm_answer.strip().lower()
        self.llm_answer = self.replace_with_yes_or_no(self.llm_answer)
        self.real_answer = self.real_answer.strip().lower()
        self.question_types = [x.strip().lower() for x in self.question_type.split(",")]
        print("LLM ANSWER:", self.llm_answer, "REAL ANSWER:", self.real_answer)

    def correct(self) -> bool:
        return self.llm_answer == self.real_answer


@dataclass
class AnswerParser:

    answer_data: List[AnswerComparison] = field(default_factory=list)

    def process_all_answers(self, test_data_file: str, answers_file: str):
        with open(test_data_file, "r") as f:
            test_data = json.load(f)

        with jsonlines.open(answers_file) as reader:
            for i, answer_data in enumerate(reader):
                correct_answer_data = test_data[i]
                question = answer_data["prompt"]
                llm_answer = answer_data["text"]
                question_type = correct_answer_data["question_type"]
                correct_answer = correct_answer_data["conversations"][1]["value"]
                self.answer_data.append(
                    AnswerComparison(
                        question=question,
                        llm_answer=llm_answer,
                        real_answer=correct_answer,
                        question_type=question_type,
                    )
                )

        question_types = set([x for d in self.answer_data for x in d.question_types])

        answer_results = {t: {"correct": 0, "total": 0} for t in question_types}
        answer_results["mean"] = {"correct": 0, "total": 0}
        for d in self.answer_data:
            for t in d.question_types:
                answer_results[t]["correct"] += d.correct()
                answer_results[t]["total"] += 1
            answer_results["mean"]["correct"] += d.correct()
            answer_results["mean"]["total"] += 1

        print("ANSWER RESULTS:", answer_results)
        # Create a bar chart of the answer results
        fig, ax = plt.subplots(figsize=(10, 6))
        question_types = list(answer_results.keys())
        correct_rates = [
            answer_results[t]["correct"] / answer_results[t]["total"]
            for t in question_types
        ]

        ax.bar(question_types, correct_rates)
        ax.set_xlabel("Question Type", fontsize=14)  # Increase the font size
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_title("Finetuned model accuracy across question types", fontsize=16)

        # Rotate the x-axis labels for better visibility
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # Make the legend more visible
        # ax.legend(loc="upper right", fontsize=12)

        # Save the plot to a file
        plt.savefig(
            "answer_results.png", dpi=300
        )  # Increase the DPI for a higher resolution image


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--test_data_file",
        type=str,
        required=True,
        help="Json file with test data and answers",
    )
    ap.add_argument(
        "--answers_file",
        type=str,
        required=True,
        help="Jsonl file with output answer data",
    )
    args = ap.parse_args()
    p = AnswerParser()
    p.process_all_answers(
        test_data_file=args.test_data_file, answers_file=args.answers_file
    )
