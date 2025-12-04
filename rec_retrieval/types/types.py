from collections import namedtuple

__all__ = [
    "UserID",
    "ItemID",
    "UserASIN",
    "ItemASIN",
    "Sequence",
    "MetaData",
    "UMap",
    "SMap",
    "TokenizedItem",
    "TokenizedSequence",
]

type UserID = int
type ItemID = int
type UserASIN = str
type ItemASIN = str

type Sequence = dict[UserID, list[ItemID]]
type MetaData = dict[ItemASIN, dict[str, str]]

type UMap = dict[UserASIN, UserID]
type SMap = dict[ItemASIN, ItemID]

TokenizedItem = namedtuple("TokenizedItem", ["input_ids", "token_type_ids", "attr_type_ids"])
TokenizedSequence = namedtuple(
    "TokenizedSequence",
    ["input_ids", "token_type_ids", "attr_type_ids", "item_position_ids", "attention_mask", "global_attention_mask"],
)
