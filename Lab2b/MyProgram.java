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

import mpi.*;

public class MyProgram
{
	private static void main_rank0() throws Exception
	{
		double array[] = new double[100];

		for (int i = 0; i < array.length; ++i)
			array[i] = Math.random() * 100;

		System.out.printf("Pre:  [%f", array[0]);

		for (int i = 1; i < array.length; ++i)
			System.out.printf(", %f", array[i]);

		System.out.println("]");

		for (int i = 1; i <= 3; ++i)
			MPI.COMM_WORLD.Send(array, i * 25, 25, MPI.DOUBLE, i, 0);

		for (int i = 0; i < 25; ++i) array[i] = Math.sqrt(array[i]);

		for (int i = 1; i <= 3; ++i)
			MPI.COMM_WORLD.Recv(array, i * 25, 25, MPI.DOUBLE, i, 0);

		System.out.printf("Post: [%f", array[0]);

		for (int i = 1; i < array.length; ++i)
			System.out.printf(", %f", array[i]);

		System.out.println("]");
	}

	private static void main_rankN() throws Exception
	{
		double array[] = new double[25];

		MPI.COMM_WORLD.Recv(array, 0, array.length, MPI.DOUBLE, 0, 0);

		for (int i = 0; i < array.length; ++i) array[i] = Math.sqrt(array[i]);

		MPI.COMM_WORLD.Send(array, 0, array.length, MPI.DOUBLE, 0, 0);
	}

	public static void main(String[] args) throws Exception
	{
		MPI.Init(args);

		int rank = MPI.COMM_WORLD.Rank();
		int size = MPI.COMM_WORLD.Size();

		if (size < 4) {
			System.err.printf("err: needs 4 machines: only %d found\n", size);
			System.exit(1);
		}

		System.out.printf("Rank: %d\n", rank);

		if (rank == 0)
			main_rank0();
		else
			main_rankN();

		MPI.Finalize();
	}
}
