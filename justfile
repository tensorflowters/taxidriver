set dotenv-load := true
set positional-arguments := true

[no-exit-message]
@venvinfo:
  poetry env info

[no-exit-message]
@venvpath:
  poetry env list

[no-exit-message]
@activate:
  #!/usr/bin/env bash
  source $(poetry env info --path)/bin/activate

[no-exit-message]
@install:
  poetry install

[no-exit-message]
@add package group='main':
  poetry add {{package}} --group={{group}}

[no-exit-message]
@rm package='':
  poetry remove {{package}}