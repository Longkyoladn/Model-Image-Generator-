"""Sổ ghi lỗi và cảnh báo phát hiện trong lúc tổng hợp."""

from __future__ import annotations


class IssueLog:
    """Danh sách (mức, thông báo) theo đúng thứ tự phát hiện.

    `for_dataset(name)` trả về một sổ dùng chung danh sách đó nhưng tự thêm
    `[name]` vào đầu mỗi thông báo.
    """

    def __init__(self, dataset: str | None = None, items: list[tuple[str, str]] | None = None) -> None:
        self.dataset = dataset
        self.items: list[tuple[str, str]] = [] if items is None else items

    def for_dataset(self, dataset: str) -> IssueLog:
        return IssueLog(dataset, self.items)

    def add(self, level: str, message: str) -> None:
        prefix = f"[{self.dataset}] " if self.dataset else ""
        self.items.append((level, prefix + message))

    def error(self, message: str) -> None:
        self.add("error", message)

    def warning(self, message: str) -> None:
        self.add("warning", message)

    @property
    def errors(self) -> list[str]:
        return [m for level, m in self.items if level == "error"]

    @property
    def warnings(self) -> list[str]:
        return [m for level, m in self.items if level == "warning"]
