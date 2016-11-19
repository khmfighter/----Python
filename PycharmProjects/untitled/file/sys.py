import sys
print type(sys.stdin)
print sys.stdin.mode
print sys.stdin.fileno()
print sys.stdout.mode
sys.stdout.write('1000')
print sys.stdout.fileno()
print sys.stderr.mode
sys.stderr.write('10000')
print sys.stdout.fileno()
