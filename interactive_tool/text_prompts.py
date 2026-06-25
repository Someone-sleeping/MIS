# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedTextPrompt:
    raw_text: str
    canonical_text: str
    target_class: str
    intent: str


CLASS_ALIASES = {
    "ceiling": ["ceiling", "天花板", "吊顶"],
    "floor": ["floor", "地板", "地面"],
    "wall": ["wall", "墙", "墙面", "墙壁"],
    "beam": ["beam", "横梁", "梁"],
    "column": ["column", "柱", "柱子"],
    "window": ["window", "窗", "窗户"],
    "door": ["door", "门", "房门"],
    "table": ["table", "desk", "桌", "桌子", "办公桌", "会议桌"],
    "chair": ["chair", "椅", "椅子", "座椅"],
    "sofa": ["sofa", "沙发"],
    "bookcase": ["bookcase", "bookshelf", "书架", "书柜"],
    "board": ["board", "whiteboard", "blackboard", "白板", "黑板", "板"],
    "clutter": ["clutter", "杂物", "其他", "杂乱物体"],
}


INTENT_ALIASES = {
    "refine": ["优化", "修正", "改进", "细化", "refine", "improve", "fix", "correct"],
    "segment": ["分割", "标出", "选中", "提取", "segment", "select", "mask"],
    "remove": ["去掉", "删除", "排除", "不要", "remove", "exclude"],
}


def parse_text_prompt(text):
    raw_text = (text or "").strip()
    normalized = raw_text.lower()

    target_class = "object"
    for class_name, aliases in CLASS_ALIASES.items():
        if any(alias.lower() in normalized for alias in aliases):
            target_class = class_name
            break

    intent = "segment"
    for intent_name, aliases in INTENT_ALIASES.items():
        if any(alias.lower() in normalized for alias in aliases):
            intent = intent_name
            break

    canonical_text = target_class if target_class != "object" else (raw_text or "object")
    return ParsedTextPrompt(
        raw_text=raw_text or canonical_text,
        canonical_text=canonical_text,
        target_class=target_class,
        intent=intent,
    )
