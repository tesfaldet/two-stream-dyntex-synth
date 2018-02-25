#!/bin/bash

REGEX=$1
OUTPUT=$2

convert -delay 10 -loop 0 -alpha set -dispose previous `ls -v $REGEX` +repage $OUTPUT
