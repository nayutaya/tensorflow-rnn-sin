
import sys
import yaml # sudo pip3 install pyyaml

if len(sys.argv) <= 1:
    print("Usage: " + sys.argv[0] + " param", file=sys.stderr)
    exit(1)

param_path = sys.argv[1]
print("param_path = %s" % param_path)

with open(param_path, "r") as file:
    param = yaml.load(file.read())

for key in param.keys():
    print("%s = %s" % (key, str(param[key])))
