"""Решение: Unsloth + Phi-3.5-mini-instruct.

На входе:  /workspace/input.pickle
На выходе: /workspace/output.json
Веса:      /workspace/weights (предварительно скачиваются download_weights.py)
"""
import json
import os
import pickle
import torch

from unsloth import FastLanguageModel

MODEL_DIR = "./weights"
MAX_NEW_TOKENS = 1024
MAX_SEQ_LENGTH = 4096


def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)
    print(rows)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    result = []
    for row in rows:
        messages = [{"role": "user", "content": row["question"]}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            temperature=0.0, # Equivalent to sampling temperature 0.0
            do_sample=False,
        )

        # Decode only the generated part
        generated_ids = outputs[0][inputs.shape[-1]:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(answer)
        result.append({"rid": row["rid"], "answer": answer.strip()})

    with open("output.json", "w") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
