from InquirerPy import get_style, inquirer
from InquirerPy.base.control import Choice

colors = {
    "answer": "#ff00fb",
    "answered_question": "",
    "answermark": "#ff0055",
    "checkbox": "#00ff7b",
    "fuzzy_border": "",
    "fuzzy_info": "#abb2bf",
    "fuzzy_match": "#ff00fb",
    "fuzzy_prompt": "#ff00fb",
    "input": "#00ff7b",
    "instruction": "#abb2bf",
    "long_instruction": "#abb2bf",
    "marker": "#ff0055",
    "pointer": "#ff00fb",
    "question": "",
    "questionmark": "#ff0055 bold",
    "separator": "",
    "skipped": "#5c6370",
    "spinner_pattern": "#ff0055",
    "spinner_text": "",
    "validator": "",
}


def select_prompt(
    choices: list[Choice], message: str, instruction: str = "", **kwargs
) -> str:
    style = get_style(
        colors,
        style_override=True,
    )
    return inquirer.select(
        choices=choices,
        message=message,
        style=style,
        instruction=instruction,
        **kwargs,
    ).execute()


def default_prompt(message: str, **kwargs) -> str:
    style = get_style(
        colors,
        style_override=True,
    )
    default = kwargs.get("default")
    return inquirer.text(
        message=f"{message}",
        style=style,
        default=f"{default}",
    ).execute()


def default_prompt_float(message: str, number: float, **kwargs) -> float:
    style = get_style(
        colors,
        style_override=True,
    )

    default = number

    return inquirer.number(
        message=f"{message}", style=style, default=default, float_allowed=True
    ).execute()
