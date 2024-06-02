import json
import argparse
from typing import Sequence, TypedDict
from dataclasses import dataclass

import jsonlines


@dataclass
class AnswerComparison:
    question: str
    llm_answer: str
    real_answer: str

    def __post_init__(self):
        self.llm_answer = self.llm_answer.strip().lower()
        self.real_answer = self.real_answer.strip().lower()

    def correct(self) -> bool:
        return self.llm_answer == self.real_answer


@dataclass
class AnswerParser:

    answer_data: Sequence[AnswerComparison] = []

    def add_answer(self, llm_answer: str, real_answer: str, question: str):
        self.answer_data.append(
            AnswerComparison(
                question=question, llm_answer=llm_answer, real_answer=real_answer
            )
        )

    def process_all_answers(self, test_data_file: str, answers_file: str):
        with open(test_data_file, "r") as f:
            test_data = json.load(f)

        with jsonlines.open(answers_file) as reader:
            for i, answer_data in enumerate(reader):
                correct_answer_data = test_data[i]
                question = answer_data["prompt"]
                llm_answer = answer_data["text"]
                correct_answer = correct_answer_data["conversations"][1]["value"]
                self.add_answer(llm_answer, correct_answer, question=question)

        total = len(self.answer_data)
        correct = sum([d.correct for d in self.answer_data])
        print("TOTAL:", total)
        print("CORRECT:", correct)


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
