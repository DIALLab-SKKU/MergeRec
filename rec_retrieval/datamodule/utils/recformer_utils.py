from transformers import BatchEncoding, PreTrainedTokenizer

from ...types import TokenizedItem, TokenizedSequence

__all__ = [
    "tokenize_item",
    "concat_tokenized_items",
    "pad_tokenized_sequences",
]


def tokenize_item(
    item_metadata: dict[str, str],
    tokenizer: PreTrainedTokenizer,
    attr_name_id_map: dict[str, int],
    max_attribute_len: int,
) -> TokenizedItem:
    all_input_ids = []
    all_token_type_ids = []
    all_attr_type_ids = []
    for k, v in item_metadata.items():
        # Tokenize value and truncate to max attribute length
        assert isinstance(v, str), "Item metadata value must be a string"

        key_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(k))
        value_tokenized = tokenizer.tokenize(v)
        value_tokenized = value_tokenized[:max_attribute_len]
        value_tokenized = tokenizer.convert_tokens_to_ids(value_tokenized)

        input_ids = key_tokenized + value_tokenized
        token_type_ids = [1] * len(key_tokenized) + [2] * len(value_tokenized)
        attr_type_ids = [attr_name_id_map[k]] * len(input_ids)

        all_input_ids.extend(input_ids)
        all_token_type_ids.extend(token_type_ids)
        all_attr_type_ids.extend(attr_type_ids)

    return TokenizedItem(
        input_ids=all_input_ids,
        token_type_ids=all_token_type_ids,
        attr_type_ids=all_attr_type_ids,
    )


def concat_tokenized_items(tokenized_items: list[TokenizedItem], bos_token_id: int) -> TokenizedSequence:
    input_ids = [bos_token_id]
    token_type_ids = [0]
    attr_type_ids = [0]
    item_position_ids = [0]
    attention_mask = [1]
    global_attention_mask = [1]

    for item_position, tokenized_item in enumerate(tokenized_items, start=1):
        input_ids += tokenized_item.input_ids
        token_type_ids += tokenized_item.token_type_ids
        attr_type_ids += tokenized_item.attr_type_ids
        item_position_ids += [item_position] * len(tokenized_item.input_ids)
        attention_mask += [1] * len(tokenized_item.input_ids)
        global_attention_mask += [0] * len(tokenized_item.input_ids)

    return TokenizedSequence(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attr_type_ids=attr_type_ids,
        item_position_ids=item_position_ids,
        attention_mask=attention_mask,
        global_attention_mask=global_attention_mask,
    )


def pad_tokenized_sequences(
    tokenized_sequences: list[TokenizedSequence],
    pad_token_id: int,
    max_length: int,
    pad_to_multiple_of: int | None = None,
) -> BatchEncoding:
    max_seq_length = max(len(seq.input_ids) for seq in tokenized_sequences)
    pad_length = min(max_seq_length, max_length)

    if pad_to_multiple_of and pad_to_multiple_of > 0:
        if pad_length % pad_to_multiple_of != 0:
            pad_length = ((pad_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            pad_length = min(pad_length, max_length)

    padded_input_ids = []
    padded_token_type_ids = []
    padded_attr_type_ids = []
    padded_item_position_ids = []
    padded_attention_mask = []
    padded_global_attention_mask = []

    for seq in tokenized_sequences:
        truncated = seq.input_ids[:pad_length]
        length_to_pad = pad_length - len(truncated)

        padded_input_ids.append(truncated + [pad_token_id] * length_to_pad)
        padded_token_type_ids.append(seq.token_type_ids[:pad_length] + [3] * length_to_pad)
        padded_attr_type_ids.append(seq.attr_type_ids[:pad_length] + [0] * length_to_pad)
        padded_item_position_ids.append(seq.item_position_ids[:pad_length] + [0] * length_to_pad)
        padded_attention_mask.append(seq.attention_mask[:pad_length] + [0] * length_to_pad)
        padded_global_attention_mask.append(seq.global_attention_mask[:pad_length] + [0] * length_to_pad)

    return BatchEncoding(
        data=dict(
            input_ids=padded_input_ids,
            token_type_ids=padded_token_type_ids,
            attr_type_ids=padded_attr_type_ids,
            item_position_ids=padded_item_position_ids,
            attention_mask=padded_attention_mask,
            global_attention_mask=padded_global_attention_mask,
        ),
        tensor_type="pt",
    )
