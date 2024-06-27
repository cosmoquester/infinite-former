from collections import Counter
import json
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_max", type=int, default=20)
parser.add_argument("--length", type=int, default=32000)
parser.add_argument("--n_train_examples", type=int, default=80000)
parser.add_argument("--n_dev_examples", type=int, default=800)
parser.add_argument("--n_test_examples", type=int, default=800)
parser.add_argument("--output_path", type=str, default="./data.json")

output_path = "./data.json"


def generate_example(length: int, sep: int, n_max: int):
    prmopt_ids = []
    p_initial = np.random.randint(1, 10, 20)
    p_initial = p_initial / p_initial.sum()
    p_final = np.random.randint(1, 10, 20)
    p_final = p_final / p_final.sum()

    for j in range(length):
        r = (j + 1) / length
        p = (1 - r) * p_initial + r * p_final
        prmopt_ids.append(np.random.choice(n_max, 1, p=p)[0])
    target_ids = [item for item, _ in Counter(prmopt_ids).most_common()]

    for i in range(n_max):
        if i not in target_ids:
            target_ids.append(i)

    prmopt_ids.append(sep)
    return {
        "prmopt_ids": prmopt_ids,
        "target_ids": target_ids,
    }

def main(args: argparse.Namespace) -> None:
    print("Generating train data...")
    train_examples = [generate_example(args.length, args.n_max, args.n_max) for _ in tqdm(range(args.n_train_examples))]

    print("Generating dev data...")
    dev_examples = [generate_example(args.length, args.n_max, args.n_max) for _ in tqdm(range(args.n_dev_examples))]

    print("Generating test data...")
    test_examples = [generate_example(args.length, args.n_max, args.n_max) for _ in tqdm(range(args.n_test_examples))]

    print("Saving train data...")
    with open(args.output_path, "w") as f:
        json.dump({
            "train": train_examples,
            "dev": dev_examples,
            "test": test_examples,
            "vocab_size": args.n_max + 1,
            "prompt_length": args.length + 1,
        }, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
