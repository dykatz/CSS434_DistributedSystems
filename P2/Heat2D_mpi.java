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

public class Heat2D_mpi
{
  private static int Rank;
  private static int Peers;
  private static int Sz;
  private static int Timespan;
  private static int Warmspan;
  private static int Interval;
  private static double Rate;

  private static double[][] z;
  private static int t, p;

  private static void printz()
  {
    if (t % Interval != 0 && t != Timespan - 1)
      return;

    System.out.printf("time = %d\n", t);

    for (int y = 1; y < Sz - 1; ++y) {
      for (int x = 1; x < Sz - 1; ++x)
        System.out.printf("%d", Math.round(Math.floor(z[p][y*Sz+x] / 2)));

      System.out.println();
    }

    System.out.println();
  }

  private static void matchedges()
  {
    for (int y = 0; y < Sz; ++y) {
      z[p][y*Sz] = z[p][y*Sz+1];
      z[p][y*Sz+Sz-1] = z[p][y*Sz+Sz-2];
    }

    for (int x = 0; x < Sz; ++x) {
      z[p][x] = z[p][Sz+x];
      z[p][(Sz-1)*Sz+x] = z[p][(Sz-2)*Sz+x];
    }
  }

  private static void hottopedge()
  {
    for (int x = Sz / 3; x < Sz * 2 / 3; ++x)
      z[p][x] = 19.0f;
  }

  private static void heatdiffuse0() throws MPIException
  {
    int q = (p + 1) % 2;

    for (int i = 1; i < Peers; ++i)
      MPI.COMM_WORLD.Send(z[p], Sz*(i*(Sz-2)/Peers), Sz*((Sz-2)/Peers+2), MPI.DOUBLE, i, 0);

    for (int y = 1; y < (Sz-2)/Peers+1; ++y)
      for (int x = 1; x < Sz-1; ++x)
        z[q][y*Sz+x] = z[p][y*Sz+x]*(1-4*Rate)+Rate*(
          z[p][(y-1)*Sz+x]+z[p][(y+1)*Sz+x]+z[p][y*Sz+x-1]+z[p][y*Sz+x+1]);

    for (int i = 1; i < Peers; ++i)
      MPI.COMM_WORLD.Recv(z[q], Sz*(i*(Sz-2)/Peers+1), Sz*(Sz-2)/Peers, MPI.DOUBLE, i, 0);
  }

  private static void heatdiffuseN() throws MPIException
  {
    int q = (p + 1) % 2;

    MPI.COMM_WORLD.Recv(z[p], 0, Sz*((Sz-2)/Peers+2), MPI.DOUBLE, 0, 0);

    for (int y = 1; y < (Sz-2)/Peers+1; ++y)
      for (int x = 1; x < Sz-1; ++x)
        z[q][y*Sz+x] = z[p][y*Sz+x]*(1-4*Rate)+Rate*(
          z[p][(y-1)*Sz+x]+z[p][(y+1)*Sz+x]+z[p][y*Sz+x-1]+z[p][y*Sz+x+1]);

    MPI.COMM_WORLD.Send(z[q], Sz, Sz*(Sz-2)/Peers, MPI.DOUBLE, 0, 0);
  }

  public static void main(String args[]) throws MPIException
  {
    MPI.Init(args);

    Rank = MPI.COMM_WORLD.Rank();
    Peers = MPI.COMM_WORLD.Size();

    Sz = args.length > 1 ? Integer.parseInt(args[1]) + 2 : 102;
    Timespan = args.length > 2 ? Integer.parseInt(args[2]) : 3000;
    Warmspan = args.length > 3 ? Integer.parseInt(args[3]) : 2700;
    Interval = args.length > 4 ? Integer.parseInt(args[4]) : 500;
    Rate = args.length > 5 ? Float.parseFloat(args[5]) : 0.2f;

    assert (Sz-2)%Peers == 0 :
      "The board size must divide evenly between the peer count";

    t = 0;
    p = 0;

    z = new double[2][Sz * (Rank == 0 ? Sz : (Sz-2)/Peers+2)];

    if (Rank == 0) {
      long start = System.nanoTime();

      for (; t < Warmspan; p = (++t) % 2) {
        matchedges();
        hottopedge();
        printz();
        heatdiffuse0();
      }

      for (; t < Timespan; p = (++t) % 2) {
        matchedges();
        printz();
        heatdiffuse0();
      }

      System.out.printf("Elapsed time = %f seconds\n",
        (double)(System.nanoTime() - start) / 1000000000.0);
    } else {
      for (; t < Timespan; p = (++t) % 2)
        heatdiffuseN();
    }

    MPI.Finalize();
  }
}
