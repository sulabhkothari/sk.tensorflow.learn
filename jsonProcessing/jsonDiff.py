import json
from collections import Iterable


def flatIterateNestedNodes(item, mapping, setOfFields):
 setOfFields.add(item)
 print(item)
 if('properties' in mapping[item].keys()):
  for nested_item in mapping[item]['properties']:
   flatIterateNestedNodes(nested_item, mapping[item]['properties'], setOfFields)


person = '{"name": "Bob", "languages": ["English", "Fench"]}'
person_dict = json.loads(person)
# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
print( person_dict)
# Output: ['English', 'French']
print(person_dict['languages'])

with open('/Users/sulabhkothari/Documents/ES7Mappings.json') as f1:
 data1 = json.load(f1)

with open('/Users/sulabhkothari/Documents/ES5Mappings.json') as f2:
 data2 = json.load(f2)

print(data1)
print(data2)
listEs5 = set()
listEs7 = set()
#print(json.dumps(data1['prod_category120_fact']['mappings']['properties'], indent = 4, sort_keys=True))

es7mapping = data1['doc_name']['mappings']['properties']
es5mapping = data2['doc_name']['mappings']['doc']['properties']

for item in es7mapping:
 flatIterateNestedNodes(item, es7mapping, listEs7)

for item in es5mapping:
 flatIterateNestedNodes(item, es5mapping, listEs5)

print(len(listEs5))
print(len(listEs7))
print(len(listEs7.intersection(listEs5)))
print(len(listEs7.difference(listEs5)))
print(len(listEs5.difference(listEs7)))


with open('/Users/sulabhkothari/Downloads/ES7.json') as f3:
 data3 = json.load(f3)

def flatIterateNestedNodes_Doc(item, docum, setOfFields):
 setOfFields.add(item)

 if(isinstance(docum, Iterable) == False or isinstance(docum, str)):
  return

 for field in docum:
  flatIterateNestedNodes_Doc(field, docum[field], setOfFields)

#print(data3)
es7doc = data3['hits']['hits'][0]['_source']
nbr = set()
for item in es7doc:
 flatIterateNestedNodes_Doc(item, es7doc[item], nbr)

print(len(nbr))

