"""Scrollable three-segment event log panel for the unified visualizer."""

from __future__ import annotations


def split_event_log(
    lines: list[str] | tuple[str, ...],
) -> tuple[list[str], list[str], list[str]]:
    """Partition replay lines into system, S1, and S2 sections."""
    system: list[str] = []
    s1: list[str] = []
    s2: list[str] = []
    for line in lines:
        if line.startswith("S1 "):
            s1.append(line)
        elif line.startswith("S2 "):
            s2.append(line)
        else:
            system.append(line)
    return system, s1, s2


def apply_chrome_palette(widget, chrome) -> None:
    """Match *widget* background and text colors to a reference chrome widget."""
    from PyQt6.QtGui import QPalette

    widget.setAutoFillBackground(True)
    chrome_pal = chrome.palette()
    pal = widget.palette()
    pal.setColor(QPalette.ColorRole.Window, chrome_pal.color(QPalette.ColorRole.Window))
    pal.setColor(
        QPalette.ColorRole.WindowText,
        chrome_pal.color(QPalette.ColorRole.WindowText),
    )
    pal.setColor(QPalette.ColorRole.Text, chrome_pal.color(QPalette.ColorRole.WindowText))
    widget.setPalette(pal)


class _LogSection:
    def __init__(self, title: str, parent, *, chrome) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont
        from PyQt6.QtWidgets import QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget

        self._frame = QFrame(parent)
        apply_chrome_palette(self._frame, chrome)
        layout = QVBoxLayout(self._frame)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(4)

        self._title = QLabel(title)
        title_font = self._title.font()
        title_font.setBold(True)
        self._title.setFont(title_font)
        apply_chrome_palette(self._title, chrome)
        layout.addWidget(self._title)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        apply_chrome_palette(self._scroll, chrome)

        self._log_label = QLabel()
        self._log_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self._log_label.setWordWrap(True)
        self._log_label.setFont(QFont("Monospace", 9))
        apply_chrome_palette(self._log_label, chrome)
        self._log_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        log_host = QWidget()
        apply_chrome_palette(log_host, chrome)
        log_host_layout = QVBoxLayout(log_host)
        log_host_layout.setContentsMargins(0, 0, 0, 0)
        log_host_layout.addWidget(self._log_label)
        log_host_layout.addStretch()
        self._scroll.setWidget(log_host)
        apply_chrome_palette(self._scroll.viewport(), chrome)
        layout.addWidget(self._scroll, stretch=1)

    @property
    def widget(self):
        return self._frame

    def set_lines(self, lines: list[str]) -> None:
        from PyQt6.QtCore import QTimer
        from PyQt6.QtGui import QFontMetrics

        text = "\n".join(lines)
        self._log_label.setText(text)
        metrics = QFontMetrics(self._log_label.font())
        line_count = max(1, text.count("\n") + 1) if text else 1
        self._log_label.setMinimumWidth(
            max(200, metrics.horizontalAdvance("M") * 28)
        )
        self._log_label.setMinimumHeight(metrics.lineSpacing() * line_count + 4)

        def _scroll_to_bottom() -> None:
            bar = self._scroll.verticalScrollBar()
            bar.setValue(bar.maximum())

        QTimer.singleShot(0, _scroll_to_bottom)


class EventLogPanel:
    """Bottom event log with system, S1, and S2 sections side by side."""

    def __init__(self, parent, *, chrome) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QFrame, QSplitter, QVBoxLayout

        self._widget = QFrame(parent)
        self._widget.setObjectName("eventLogPanel")
        self._widget.setMinimumHeight(120)
        apply_chrome_palette(self._widget, chrome)

        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        self._splitter = QSplitter(Qt.Orientation.Horizontal, self._widget)
        apply_chrome_palette(self._splitter, chrome)
        self._system = _LogSection("System", self._splitter, chrome=chrome)
        self._s1 = _LogSection("S1", self._splitter, chrome=chrome)
        self._s2 = _LogSection("S2", self._splitter, chrome=chrome)
        self._splitter.addWidget(self._system.widget)
        self._splitter.addWidget(self._s1.widget)
        self._splitter.addWidget(self._s2.widget)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setStretchFactor(2, 2)
        self._splitter.setSizes([240, 480, 480])
        self._splitter.setChildrenCollapsible(False)
        layout.addWidget(self._splitter)

    @property
    def widget(self):
        return self._widget

    def set_lines(self, lines: list[str] | tuple[str, ...]) -> None:
        system, s1, s2 = split_event_log(lines)
        self._system.set_lines(system)
        self._s1.set_lines(s1)
        self._s2.set_lines(s2)
