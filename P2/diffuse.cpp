/* Copyright (C) 2018 Dylan Katz
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <stdio.h>
#include <math.h>

#define Warmtime 2700
#define Time     3000
#define Sz       100
#define Hot      19.f
#define Rate     0.2f
#define Interval 20

int
main(void)
{
	float z[2][Sz][Sz];
	int p, x, y, t;

	for (p = 0; p < 2; ++p) {
		for (x = 0; x < Sz; ++x) {
			for (y = 0; y < Sz; ++y)
				z[p][x][y] = 0.f;
		}
	}

	for (t = 0, p = 0; t < Time; p = (++t) % 2) {
		for (y = 0; y < Sz; ++y) {
			z[p][0][y] = z[p][1][y];
			z[p][Sz - 1][y] = z[p][Sz - 2][y];
		}

		for (x = 0; x < Sz; ++x) {
			z[p][x][0] = z[p][x][1];
			z[p][x][Sz - 1] = z[p][x][Sz - 2];
		}

		if (t < Warmtime) {
			for (x = Sz / 3; x < Sz * 2 / 3; ++x)
				z[p][x][0] = Hot;
		}

		if (!(t % Interval)) {
			printf("time = %d\n", t);

			for (y = 0; y < Sz; ++y) {
				for (x = 0; x < Sz; ++x) {
					printf("%ld", lrintf(floorf(
						z[p][x][y] / 2)));
				}

				printf("\n");
			}

			printf("\n");
		}

		for (x = 1; x < Sz - 1; ++x) {
			for (y = 1; y < Sz - 1; ++y) {
				z[(p+1)%2][x][y] = z[p][x][y]*(1-4*Rate)+Rate*(
					z[p][x+1][y] + z[p][x-1][y] +
					z[p][x][y+1] + z[p][x][y-1]);
			}
		}
	}
}
