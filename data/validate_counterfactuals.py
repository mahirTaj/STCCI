"""Validate only supplied MIB pairs; never construct counterfactual text."""


def validate_record(row, index, tokenizer, config):
    def require(condition, reason):
        if not condition:
            raise ValueError(reason)

    cf = row.get(config["counterfactual"])
    require(isinstance(cf, dict), "missing_counterfactual")
    for item in (row, cf):
        require(isinstance(item.get("prompt"), str) and bool(item["prompt"]), "invalid_prompt")
        choices = item.get("choices")
        require(isinstance(choices, list) and len(choices) == 2 and
                all(isinstance(s, str) and s.strip() for s in choices) and
                len(set(choices)) == 2, "ambiguous_choices")
        require(type(item.get("answerKey")) is int and item["answerKey"] in (0, 1), "invalid_answer_key")
    require(isinstance(row.get("template"), str) and bool(row["template"]), "missing_template")
    require(set(row["choices"]) == set(cf["choices"]), "incompatible_choice_sets")
    target = row["choices"][row["answerKey"]]
    distractor = row["choices"][1 - row["answerKey"]]
    require(cf["choices"][cf["answerKey"]] == distractor, "counterfactual_answer_not_flipped")
    metadata = row.get("metadata", {})
    require(metadata.get("indirect_object") == target and metadata.get("subject") == distractor,
            "inconsistent_source_metadata")
    tokens = [tokenizer.encode(item["prompt"], add_special_tokens=False) for item in (row, cf)]
    require(all(0 < len(t) <= config["max_length"] for t in tokens), "sequence_length")
    require(len(tokens[0]) == len(tokens[1]), "incompatible_prompt_lengths")
    answers = []
    for answer in (target, distractor):
        answer_tokens = tokenizer.encode(" " + answer, add_special_tokens=False)
        require(len(answer_tokens) == 1, "multi_token_answer")
        # Confirm the answer is one continuation token in BOTH actual contexts.
        for item, prompt_tokens in zip((row, cf), tokens):
            continuation = tokenizer.encode(item["prompt"] + " " + answer, add_special_tokens=False)
            require(continuation == prompt_tokens + answer_tokens, "incompatible_answer_boundary")
        answers.append(answer_tokens[0])
    require(answers[0] != answers[1], "ambiguous_answer_tokens")
    return {"record_id": f"{config['dataset_revision']}:train:{index}", "source_index": index,
            "template": row["template"], "clean_tokens": tokens[0], "corrupt_tokens": tokens[1],
            "target_token": answers[0], "distractor_token": answers[1]}
