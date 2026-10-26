
from retriever import retrieve

TEST_CASES = [
    {
        "question": "What provision covers arrest without a warrant?",
        "expected_section": "35"
    },
    {
        "question": "What provision concerns information about an arrested person's grounds of arrest?",
        "expected_section": "47"
    },
    {
        "question": "What provision concerns a police officer requiring a subordinate to make an arrest?",
        "expected_section": "55"
    }
]


def evaluate():
    correct = 0

    for test in TEST_CASES:
        results = retrieve(test["question"], top_k=5)

        sections = [
            item["metadata"].get("section_number", "")
            for item in results
        ]

        passed = test["expected_section"] in sections
        correct += passed

        print(f"\nQuestion: {test['question']}")
        print(f"Expected section: {test['expected_section']}")
        print(f"Retrieved sections: {sections}")
        print(f"Result: {'PASS' if passed else 'FAIL'}")

    print(f"\nRecall@5: {correct}/{len(TEST_CASES)}")


if __name__ == "__main__":
    evaluate()
