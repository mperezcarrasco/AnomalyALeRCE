#!/bin/bash
declare -A dictionary
dictionary["ztf"]="1Wwh6TJYq4i65tG5Y4KxFq-wfwslV0pOO"

dictionary["ztf-processed"]="1G1cqfK5jl3fF8G4y4jdGn9wfj-2_0p0U"


FILEID=${dictionary[$1]}
echo $FILEID

NAME=${1%-*}
mkdir -p ztf/
DIR=./ztf/
OUTFILE=./ztf/$NAME.zip
echo $OUTFILE

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILEID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $OUTFILE && rm -rf /tmp/cookies.txt


unzip $OUTFILE -d $DIR
rm -rf $OUTFILE
