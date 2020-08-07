import argparse
import re

def numbersFromNumberIntervalsString(s):
  elem = '((\d+)-(\d+))|(\d+)'
  if re.fullmatch(f'$|({elem})(\s*,\s*({elem}))*', s) is None:
    raise argparse.ArgumentTypeError(
      'Has to be a comma separated list of numbers and ranges in the form of <num>-<num>.')

  numbers = set()
  for find in re.findall(elem, s):
    if len(find[0]) > 0:
      numbers.update(range(int(find[1]), int(find[2]) + 1))
    else:
      numbers.add(int(find[3]))
  return sorted(numbers)