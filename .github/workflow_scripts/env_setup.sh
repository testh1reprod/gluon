function setup_build_env {
    python -m pip install --upgrade pip
    python -m pip install tox
    python -m pip install flake8
    python -m pip install bandit
    python -m pip install packaging
    python -m pip install ruff
}

function install_all {
    python -m pip install -e .[dev]
}