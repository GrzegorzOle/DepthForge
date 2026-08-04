import ast, sys

PATH = 'gimp_plugins/depthforge/depthforge.py'

# The subprocess helpers live in the plugin as string literals, so nothing in
# the toolchain — editor, import, byte-compilation — ever parses them. A typo
# there stays invisible until a run fails inside GIMP, where the only evidence
# is a SyntaxError in a temp file that has already been deleted.
EMBEDDED = ('_HELPER_SCRIPT', '_UPDATE_SCRIPT')

try:
    with open(PATH, encoding='utf-8') as f:
        src = f.read()
    tree = ast.parse(src)
except SyntaxError as e:
    print('SYNTAX_ERROR: ' + str(e))
    sys.exit(1)

print('SYNTAX_OK  ' + PATH + '  lines=' + str(src.count('\n')))

found = {}
for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
        continue
    if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
        continue
    for target in node.targets:
        if isinstance(target, ast.Name) and target.id in EMBEDDED:
            found[target.id] = node.value.value

status = 0
for name in EMBEDDED:
    body = found.get(name)
    if body is None:
        print('MISSING: ' + name + ' is not a module-level string assignment')
        status = 1
        continue
    try:
        ast.parse(body, filename='<' + name + '>')
        print('SYNTAX_OK  ' + name + '  lines=' + str(body.count('\n')))
    except SyntaxError as e:
        print('SYNTAX_ERROR in ' + name + ': ' + str(e))
        status = 1

sys.exit(status)
