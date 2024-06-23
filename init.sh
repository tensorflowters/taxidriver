#!/usr/bin/env bash

VAR_SHELL=""
VAR_SHELL_VERSION=""
VAR_PARENT_SHELL=$(ps -p $$ -ocomm=)
XSH_PATH=""
JUST_DEFAULT_DIR="~/.local/bin"
JUST_EXIST=$(dpkg -l | grep just)

if [ -n "$SHELL" ]; then
  echo "Detected shell: $SHELL"
  echo "Version: $(echo $($SHELL --version | head -n 1))"

  VAR_SHELL=$(which $SHELL)
  VAR_SHELL_VERSION=$($SHELL --version | head -n 1)
elif [ -n "$VAR_PARENT_SHELL" ]; then
  echo "Detected shell avec ps: $$VAR_PARENT_SHELL"
  echo "Version: $(echo $($$VAR_PARENT_SHELL --version | head -n 1))"

  VAR_SHELL=$(which $$VAR_PARENT_SHELL)
  VAR_SHELL_VERSION=$(echo $($$VAR_PARENT_SHELL --version | head -n 1))
elif [ -n "$BASH_VERSION" ]; then
  echo "Detected shell: Bash"
  echo "Version: $BASH_VERSION"

  VAR_SHELL=VAR_SHELL=$(which bash)
  VAR_SHELL_VERSION=$BASH_VERSION
elif [ -n "$ZSH_VERSION" ]; then
  echo "Detected shell: Zsh"
  echo "Version: $ZSH_VERSION"

  VAR_SHELL=$(which zsh)
  VAR_SHELL_VERSION=$ZSH_VERSION
elif [ -n "$KSH_VERSION" ] || [ -n "$FCEDIT" ]; then
  echo "Detected shell: Ksh"
  # KSH_VERSION n'est pas toujours disponible, FCEDIT est utilisé comme indicateur pour ksh93
  if [ -n "$KSH_VERSION" ]; then
    echo "Version: $KSH_VERSION"

    VAR_SHELL=$(which ksh)
    VAR_SHELL_VERSION=$KSH_VERSION
  else
    echo "Version: ksh93 (Version exacte non disponible)"

    VAR_SHELL=$(which ksh)
    VAR_SHELL_VERSION="ksh93"
  fi
elif [ "$0" = "dash" ] || [ "$0" = "/bin/dash" ]; then
  echo "Detected shell: Dash"
  echo "Version: Not available"

  VAR_SHELL=$(which dash)
  VAR_SHELL_VERSION=$(dash --version | head -n 1 | awk '{print $2}')

elif [ -n "$FISH_VERSION" ]; then
  echo "Detected shell: Fish"
  echo "Version: $FISH_VERSION"

  VAR_SHELL=$(which fish)
  VAR_SHELL_VERSION=$FISH_VERSION
elif [ -n "$TCSH_VERSION" ]; then
  echo "Detected shell: Tcsh"
  echo "Version: $TCSH_VERSION"

  VAR_SHELL=$(which tcsh)
  VAR_SHELL_VERSION=$TCSH_VERSION
elif [ -n "$COLUMNS" ] && [ -z "$BASH_VERSION" ] && [ -z "$ZSH_VERSION" ]; then
  # Condition très basique pour csh, qui n'a pas de variable spécifique comme bash ou zsh
  echo "Detected shell: Csh ou un shell compatible"
  echo "Version: Not available"

  VAR_SHELL="Detected shell: Csh ou un shell compatible"
  VAR_SHELL_VERSION="Not available"
  exit 1
else
  echo "Undetermined or unsupported shell"
  exit 1
fi

if [[ "$VAR_SHELL" == */bash ]]; then
  echo "Configuration pour bash."
  XSH_PATH=~/.bashrc
elif [[ "$VAR_SHELL" == */zsh ]]; then
  echo "Configuration pour zsh."
  XSH_PATH=~/.zshrc
elif [[ "$VAR_SHELL" == */ksh ]]; then
  echo "Configuration pour ksh."
  XSH_PATH=~/.kshrc
elif [[ "$VAR_SHELL" == */fish ]]; then
  echo "Configuration pour fish."
  XSH_PATH=~/.fishrc
elif [[ "$VAR_SHELL" == */tcsh ]]; then
  echo "Configuration pour tcsh."
  XSH_PATH=~/.tcshrc
elif [[ "$VAR_SHELL" == */dash ]]; then
  echo "Configuration pour dash."
  XSH_PATH=~/.dashrc
else
  echo "Ce script est conçu uniquement pour bash, zsh, ksh, fish, tcsh or dash."
  exit 1
fi

install_pipx() {
    # Install pipx if not present
    if ! command -v pipx &> /dev/null; then
        read -p "Pipx is not installed. Install it now? (y/n) " response
        if [[ $response == "y" ]]; then
            local os="`uname`"
            if [[ $os == "Linux" ]]; then
              if [ $(id -u) == 0 ]; then
                echo -e "Installing pipx...\n"
                apt update
                apt install pipx
                pipx ensurepath --global
                $VAR_SHELL -c "source $XSH_PATH || echo -e \"\n\e[31m[ERROR] - Failed to source $XSH_PATH\e[0m\""
              else
                echo -e "Installing pipx...\n"
                sudo apt update
                sudo apt install pipx
                sudo pipx ensurepath --global
                $VAR_SHELL -c "source $XSH_PATH || echo -e \"\n\e[31m[ERROR] - Failed to source $XSH_PATH\e[0m\""
              fi
            elif [[ $os == "Darwin" ]]; then
                brew install pipx
                pipx ensurepath
                if [ $(id -u) == 0 ]; then
                  pipx ensurepath --global
                else
                  sudo pipx ensurepath --global
                fi
            else
                echo -e `Failed to detect OS. Linux and Mac are the only OS handled for now.\nGo to: https://pipx.pypa.io/stable/installation/ in order to install pipx yourself.`
                exit 1
            fi
        else
            echo -e "If you still want to install pipx by yourself. Go to: https://pipx.pypa.io/stable/installation/"
            exit 1
        fi
    else
        echo "Pipx already installed"
        return 0
    fi
}

install_poetry() {
    # Check if Poetry is installed using pipx
    if command -v pipx &> /dev/null; then
        if ! pipx list | grep -q "poetry"; then
            # Prompt the user to install Poetry
            read -p "Poetry is not installed. Install it now? (y/n) " response
            if [[ $response == "y" ]]; then
                echo "Installing Poetry using pipx..."
                pipx run poetry install
                return 0
            else
                echo -e "If you still want to install poetry by yourself. Go to https://python-poetry.org/docs/#installation"
                exit 1
            fi
        else
            echo "Canno't detect any poetry installed version!"
            exit 1
        fi
    else
        echo "Poetry already installed"
        return 0
    fi
}

if [ -n "$VAR_SHELL" ]; then
  if [ -n "$XSH_PATH" ]; then
    install_pipx
    install_poetry
    if [ ! "$JUST_EXIST" ]; then
      echo -e "Installing just binary to ~/.local/bin/just by default.\nKeep it as default installation path ?[y/n]:"
      read response
      if [ "$response" == "y" ] || [ "$response" == "yes" ]; then
        if [ $(id -u) == 0 ]; then
          if [ "`uname`" == "Linux" ]; then
            apt update
            apt install curl
          else
            brew install curl
          fi
        else
          if [ "`uname`" == "Linux" ] ; then
            apt update
            apt install curl
          else
            brew install curl
          fi
        fi
        curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin || echo -e "\n\e[31m[ERROR] - Failed to curl https://just.systems/install.sh\e[0m" && exit 1
        grep -qxF "PATH=\"$JUST_DEFAULT_DIR:\$PATH\"" "$XSH_PATH" || echo "PATH=\"$JUST_DEFAULT_DIR:\$PATH\"" >>"$XSH_PATH"
        $VAR_SHELL -c "source $XSH_PATH || echo -e \"\n\e[31m[ERROR] - Failed to source $XSH_PATH\e[0m\""
        just --version
      elif [ "$response" == "n" ] || [ "$response" == "no"]; then
        echo "Please enter the path:"
        read user_path
        if [$(id -u) == 0]; then
          apt update
          apt install curl
        else
          apt update
          apt install curl
        fi
        curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to $user_path || echo -e "\n\e[31m[ERROR] - Failed to curl https://just.systems/install.sh\e[0m" && exit 1
        grep -qxF "PATH=\"$user_path:\$PATH\"" "$XSH_PATH" || echo "PATH=\"$user_path:\$PATH\"" >>"$XSH_PATH"
        $VAR_SHELL -c "source $XSH_PATH || echo -e \"\n\e[31m[ERROR] - Failed to source $XSH_PATH\e[0m\""
        just --version
      else
        echo -e "Please answer by y or n.\n\e[31m[ERROR] - Failed to install just binary!\e[0m"
        exit 1
      fi
    else
      echo -e "Already find a version of just at $(which just)"
      exit 0
    fi
  else
    echo -e "\n\e[31m[ERROR] - Failed to found your shell *rc file\e[0m"
    exit 1
  fi
else
  echo -e "\n\e[31m[ERROR] - Failed to identify your shell!\e[0m"
  exit 1
fi

echo "Initialisation failed. System variables are not available."

exit 0