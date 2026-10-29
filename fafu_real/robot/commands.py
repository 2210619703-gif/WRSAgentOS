"""终端里的一句中文 → 整段煮面，或其中一步。不接大模型，只对词。"""

from __future__ import annotations

import re

# 这些词表示把 TASK_SEQUENCE 整段做完。先判断，避免「制作食物」被当成抓食物。
_FULL = ("制作食物", "制作一下", "帮我制作", "帮我做", "煮面", "全程", "全部流程", "做完")
_QUIT = ("退出", "结束", "quit", "exit", "q")

# 短语长的优先。同一次运行里可以一步一步说，手里拿着什么会留着。
_STEPS = (
    (
        ("面条放进漏勺", "食物放进漏勺", "面放进漏勺", "把面放进漏勺"),
        "seat_in",
        {"object": "noodles", "container": "strainer"},
    ),
    (
        ("漏勺放进锅", "漏勺放到锅", "漏勺入锅", "把漏勺放进锅"),
        "seat_in",
        {"object": "strainer", "container": "pot"},
    ),
    (
        ("倒到盘子", "倒进盘子", "倒入盘子", "倒到盘里"),
        "pour",
        {"source": "strainer", "target": "plate"},
    ),
    (
        ("漏勺放回桌上", "漏勺放到桌上", "放回桌上", "把漏勺放下"),
        "place_on",
        {"object": "strainer", "surface": "table"},
    ),
    (("沥水", "抖一抖", "抖掉水"), "shake_off", {"object": "strainer"}),
    (("捞出漏勺", "从锅里拿出", "漏勺出锅"), "pick", {"object": "strainer"}),
    (("抓面条", "拿面条", "取面条", "抓食物", "拿食物"), "pick", {"object": "noodles"}),
    (("抓漏勺", "拿漏勺", "取漏勺", "夹漏勺"), "pick", {"object": "strainer"}),
    (("回家", "回初始", "回到初始"), "move_named_pose", {"pose": "home"}),
)


def help_text() -> str:
    return (
        "整段：帮我制作一下食物\n"
        "一步：抓食物、食物放进漏勺、抓漏勺、漏勺放进锅、捞出漏勺、沥水、倒到盘子、漏勺放回桌上、回家\n"
        "退出：q"
    )


def parse_command(text: str):
    """quit、sequence、skill、unknown、ambiguous 五种结果。"""
    raw = (text or "").strip()
    compact = re.sub(r"\s+", "", raw).lower()
    if not compact:
        return ("empty",)
    if compact in _QUIT:
        return ("quit",)
    if any(key in compact for key in _FULL):
        return ("sequence",)
    hits = []
    for phrases, name, args in _STEPS:
        for phrase in phrases:
            start = compact.find(phrase)
            if start < 0:
                continue
            hits.append((start, start + len(phrase), name, args))
    if not hits:
        return ("unknown",)
    hits.sort(key=lambda item: item[1] - item[0], reverse=True)
    best = hits[0]
    for start, end, name, _args in hits[1:]:
        overlaps = start < best[1] and end > best[0]
        if name != best[2] and not overlaps:
            return ("ambiguous",)
    return ("skill", best[2], dict(best[3]))
