#!/bin/bash
#Author: Michiel Stock
#turns image in trace bitmap

convert "$1" pgm: | potrace -o "$2"
