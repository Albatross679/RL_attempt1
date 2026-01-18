#!/home/turn_cloak/RL_attempt1/.venv/bin/python3
import sys
from clang_format import clang_format_diff
if __name__ == '__main__':
    if sys.argv[0].endswith('.exe'):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(clang_format_diff())
