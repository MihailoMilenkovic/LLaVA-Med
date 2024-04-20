import json
import argparse
import jsonlines


def compare_answers(llm_answer: str, real_answer: str, question: str):
    print("QUESTION:", question)
    print("LLM ANSWER:", llm_answer)
    print("REAL ANSWER:", real_answer)


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
    with open(args.test_data_file, "r") as f:
        test_data = json.load(f)
    with jsonlines.open(args.answers_file) as reader:
        for i, answer_data in enumerate(reader):
            correct_answer_data = test_data[i]
            question = answer_data["prompt"]
            llm_answer = answer_data["text"]
            correct_answer = correct_answer_data["conversations"][1]["value"]
            compare_answers(llm_answer, correct_answer, question=question)
