#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
easse evaluate -t turk -m 'bleu,sari,fkgl' -q < "$SCRIPT_DIR/data/system_outputs/turk/lower/DMASS-DCSS.tok.low"
easse report -t turk -m 'bleu,sari,fkgl' < "$SCRIPT_DIR/data/system_outputs/turk/lower/DMASS-DCSS.tok.low"