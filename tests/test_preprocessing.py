import pytest

from data.fixed_population import choose_population
from data.validate_counterfactuals import validate_record


def test_official_counterfactual_and_answer_tokens(official_rows, tokenizer, config):
    for i, row in enumerate(official_rows):
        result = validate_record(row, i, tokenizer, config)
        assert result["target_token"] != result["distractor_token"]
        assert len(result["clean_tokens"]) == len(result["corrupt_tokens"])
        assert tokenizer.decode([result["target_token"]]) == " " + row["choices"][row["answerKey"]]
    assert official_rows[0]["answerKey"] == 0
    assert official_rows[0][config["counterfactual"]]["answerKey"] == 1


def test_population_is_order_independent(official_rows, tokenizer, config):
    records = [validate_record(row, i, tokenizer, config) for i, row in enumerate(official_rows)]
    assert choose_population(records, 2) == choose_population(records[::-1], 2)
    assert len({r["record_id"] for r in choose_population(records, 3)}) == 3


def test_population_refuses_missing_records(official_rows, tokenizer, config):
    records = [validate_record(row, i, tokenizer, config) for i, row in enumerate(official_rows)]
    with pytest.raises(ValueError, match="need 1024"):
        choose_population(records, 1024)


def test_actual_templates_get_round_robin_turns(official_rows, tokenizer, config):
    records = [validate_record(row, i, tokenizer, config) for i, row in enumerate(official_rows)]
    selected = choose_population(records, len(records))
    unique_templates = len({r["template"] for r in records})
    assert len({r["template"] for r in selected[:unique_templates]}) == unique_templates


def test_length_exclusion_on_real_record(official_rows, tokenizer, config):
    # Tightening an input validation bound is a unit test, never a frozen study config.
    with pytest.raises(ValueError, match="sequence_length"):
        validate_record(official_rows[0], 0, tokenizer, {**config, "max_length": 1})


def test_missing_counterfactual_is_rejected(tokenizer, config):
    # Malformed empty input, not a fabricated research record.
    with pytest.raises(ValueError, match="missing_counterfactual"):
        validate_record({}, 0, tokenizer, config)
