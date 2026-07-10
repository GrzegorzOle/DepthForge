import ast, sys
try:
    with open('gimp_plugins/depthforge/depthforge.py', encoding='utf-8') as f:
        src = f.read()
    ast.parse(src)
    print('SYNTAX_OK  lines=' + str(src.count('\n')))
except SyntaxError as e:
    print('SYNTAX_ERROR: ' + str(e))
    sys.exit(1)

