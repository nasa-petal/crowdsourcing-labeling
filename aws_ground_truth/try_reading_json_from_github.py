import pandas as pd

df = pd.read_json("https://raw.githubusercontent.com/nasa-petal/data-collection-and-prep/main/golden.json")
# df = pd.read_json("golden.json")

df.head()
#
# import json
# with open('gdb.json') as datafile:
#     data = json.load(datafile)
# retail = pd.DataFrame(data)
