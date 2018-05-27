#!/bin/sh
#
# Copyright (C) 2018 Dylan Katz
#
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

make_part2()
{
	nvcc -std=c++11 -o _part2 part2.cu
}

run_part2()
{
	MAX_BLOCKS=$1
	MAX_THREADS=$2

	for b in `seq 10 10 $MAX_BLOCKS`
	do
		for t in `seq 10 10 $MAX_THREADS`
		do
			echo "Op B=$b T=$t"
			./_part2 -b $b -t $t > output_${b}_${t}.txt
		done
	done
}

make_data()
{
	find . -type f -name '*.txt' -print0 | xargs -0 -n1 -P4 gawk '
		match($0, /Results for B=([0-9]+), T=([0-9]+)/, a) {B=a[1];T=a[2]}
		match($0, /Total elapsed time = ([0-9\.eE\-]+)/, a) {Total=a[1]}
		match($0, /Operation time = ([0-9\.eE\-]+)/, a) {Op=a[1]}
		match($0, /GPU memory allocation time = ([0-9\.eE\-]+)/, a) {Mal=a[1]}
		match($0, /GPU memory free time = ([0-9\.eE\-]+)/, a) {Fre=a[1]}
		END {print B "," T "," Mal "," Fre "," Op "," Total}
	' | awk '
	BEGIN {print "B,T,Mal,Fre,Op,Total"}{print $0}'
}
