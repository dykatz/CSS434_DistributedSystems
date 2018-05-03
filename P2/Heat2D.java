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

public class Heat2D
{
  private static int Sz;
  private static int Timespan;
  private static int Warmspan;
  private static int Interval;
  private static float Rate;

  private static float[][][] z;
  private static int t, p, y, x;

  private static void printz()
  {
    if (t % Interval != 0 && t != Timespan - 1)
      return;

    System.out.printf("time = %d\n", t);

    for (y = 1; y < Sz - 1; ++y) {
      for (x = 1; x < Sz - 1; ++x)
        System.out.printf("%d", Math.round(Math.floor(z[p][x][y] / 2)));

      System.out.println();
    }

    System.out.println();
  }

  private static void matchedges()
  {
    for (y = 0; y < Sz; ++y) {
      z[p][0][y] = z[p][1][y];
      z[p][Sz - 1][y] = z[p][Sz - 2][y];
    }

    for (x = 0; x < Sz; ++x) {
      z[p][x][0] = z[p][x][1];
      z[p][x][Sz - 1] = z[p][x][Sz - 2];
    }
  }

  private static void hottopedge()
  {
    for (x = Sz / 3; x < Sz * 2 / 3; ++x)
      z[p][x][0] = 19.0f;
  }

  private static void heatdiffuse()
  {
    int q = (p + 1) % 2;

    for (x = 1; x < Sz - 1; ++x) {
      for (y = 1; y < Sz - 1; ++y) {
        z[q][x][y] = z[p][x][y]*(1 - 4*Rate) + Rate*(
          z[p][x-1][y] + z[p][x+1][y] + z[p][x][y-1] + z[p][x][y+1]);
      }
    }
  }

  public static void main(String args[])
  {
    Sz = args.length > 0 ? Integer.parseInt(args[0]) + 2 : 102;
    Timespan = args.length > 1 ? Integer.parseInt(args[1]) : 3000;
    Warmspan = args.length > 2 ? Integer.parseInt(args[2]) : 2700;
    Interval = args.length > 3 ? Integer.parseInt(args[3]) : 500;
    Rate = args.length > 4 ? Float.parseFloat(args[4]) : 0.2f;

    z = new float[2][Sz][Sz];
    t = 0;
    p = 0;
    y = 0;
    x = 0;

    long start = System.nanoTime();

    for (; t < Warmspan; p = (++t) % 2) {
      matchedges();
      hottopedge();
      printz();
      heatdiffuse();
    }

    for (; t < Timespan; p = (++t) % 2) {
      matchedges();
      printz();
      heatdiffuse();
    }

    System.out.printf("Elapsed time = %d\n",
      (double)(System.nanoTime() - start) / 1000000000.0);
  }
}
