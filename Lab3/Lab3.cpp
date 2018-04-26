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
#include <stdlib.h>
#include <pthread.h>
#include <err.h>

int		 turn, n;
pthread_mutex_t	 lock;
pthread_cond_t	*cond;

int		 start_program(void);
void		*start_thread(void *);

int
main(int argc, char *argv[])
{
	if (argc != 2)
		errx(1, "needs one argument (thread count)");

	if ((n = atoi(argv[1])) < 1)
		errx(1, "thread count must be ≥ 1");

	return start_program();
}

int
start_program(void)
{
	pthread_t	tid[n];
	size_t		i;

	turn = 0;
	pthread_mutex_init(&lock, NULL);

	if (!(cond = (pthread_cond_t *)calloc(n, sizeof(pthread_cond_t))))
		err(1, "calloc");

	for (i = 0; i < n; ++i) {
		pthread_cond_init(cond + i, NULL);
		pthread_create(tid + i, NULL, start_thread, (void *)i);
	}

	for (i = 0; i < n; ++i) {
		pthread_join(tid[i], NULL);
		pthread_cond_destroy(cond + i);
	}

	free(cond);
	pthread_mutex_destroy(&lock);
	return 0;
}

void *
start_thread(void *arg)
{
	size_t	i, loop;

	i = (size_t)arg;

	for (loop = 0; loop < 10; ++loop) {
		printf("thread %lu iteration %lu\n", i, loop);
		pthread_mutex_lock(&lock);

		while (turn != i)
			pthread_cond_wait(cond + i, &lock);

		pthread_cond_signal(cond + (turn = (turn + 1) % n));
		pthread_mutex_unlock(&lock);
	}

	return NULL;
}
