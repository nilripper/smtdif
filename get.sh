#!/bin/sh
# get.sh url size out : fetch out, of size bytes, from url, a range at a time, side by side.
# a server that limits each connection to a rate then gives that rate times the number of ranges.
# a range cut short is taken up where it stopped, so the script may simply be rerun.
set -e
url=$1
size=$2
out=$3
n=${GETPARTS:-8}
chunk=$(( (size + n - 1) / n ))
i=0
while [ $i -lt $n ]; do
	s=$(( i * chunk ))
	e=$(( s + chunk - 1 ))
	[ $e -ge $size ] && e=$(( size - 1 ))
	part="$out.part$i"
	have=0
	[ -e "$part" ] && have=$(wc -c < "$part")
	if [ $have -ne $(( e - s + 1 )) ]; then
		curl -sS -L --retry 5 --retry-delay 3 -r "$(( s + have ))-$e" "$url" >> "$part" &
	fi
	i=$(( i + 1 ))
done
wait
# wait tells nothing of a range that failed in the background: every part is weighed
# before any is joined, and out is left as it was while one came short
i=0
while [ $i -lt $n ]; do
	s=$(( i * chunk ))
	e=$(( s + chunk - 1 ))
	[ $e -ge $size ] && e=$(( size - 1 ))
	have=$(wc -c < "$out.part$i" 2> /dev/null || echo 0)
	if [ "$have" -ne $(( e - s + 1 )) ]; then
		echo "get.sh: part $i of $out has $have of $(( e - s + 1 )) bytes; rerun to take it up" >&2
		exit 1
	fi
	i=$(( i + 1 ))
done
i=0
while [ $i -lt $n ]; do
	cat "$out.part$i" >> "$out.whole"
	i=$(( i + 1 ))
done
mv "$out.whole" "$out"
rm -f "$out".part*
[ "$(wc -c < "$out")" -eq "$size" ] || { echo "get.sh: $out is not of $size bytes" >&2; exit 1; }
