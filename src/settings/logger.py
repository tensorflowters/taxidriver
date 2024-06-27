"""
**Summary :**
  Logging module so that we can log messages using rich library with one time setup.
"""

from typing import Any, Optional

from rich import print
from rich.panel import Panel
from rich.pretty import Pretty


def p(
    *content: Any,
    title: Optional[str] = 'At least write "coucou" dumbass',
) -> None:
    """
    **Summary**: Panel print from rich library so it easier to read.
    **Args**:
        title (Optional[str]): Title of the panel.
        msg (Optional[str]): Custom message to identified the panel.
        content (Any): Content to be printed.
    **Returns**: None
    """

    pretty = Pretty(content, expand_all=True, insert_line=True)
    panel = Panel(pretty, title=title, highlight=True)

    print(panel, sep="\n")
