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

package Lab1b;

import java.net.*;
import java.io.*;

public class TcpServerDouble
{
	public static void main(String args[]) throws Exception
	{
		if (args.length != 2) {
			System.err.println("err: needs two arguments");
			System.err.println("usage: ... port size");
			return;
		}

		try {
			ServerSocket svr = new ServerSocket(Integer.parseInt(args[0]));
			byte multiplier = 1;
			int size = Integer.parseInt(args[1]);

			for (;;) {
				Socket socket = svr.accept();
				ObjectInputStream in = new ObjectInputStream(socket.getInputStream());
				ObjectOutputStream out = new ObjectOutputStream(socket.getOutputStream());
				
				Double[] data = (Double[])in.readObject();

				for (int i = 0; i < data.length; ++i)
					data[i] *= multiplier;

				out.writeObject((Object)data);
				socket.close();
				multiplier *= 2;
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
}
