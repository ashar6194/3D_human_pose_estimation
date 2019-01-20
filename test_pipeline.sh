#!/bin/bash
for number in {1..12}
do
python test.py --subject $number
done
exit 0