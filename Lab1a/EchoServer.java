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

package Lab1a;

import java.net.*;
import java.io.*;

public class EchoServer
{
	public static void main(String args[])
	{
		if (args.length != 1) {
			System.err.println("usage: java EchoServer port");
			System.exit(1);
		}

		try {
			ServerSocket server = new ServerSocket(Integer.parseInt(args[0]));

			while (true) {
				Socket client = server.accept();

				InetAddress addr = client.getInetAddress();
				System.out.printf("New connection from %s (%s)\n", addr.getHostName(), addr.getHostAddress());

				InputStream input = client.getInputStream();
				OutputStream output = client.getOutputStream();
				int amount_read = 0;
				byte buffer[] = new byte[1024];

				do {
					amount_read = input.read(buffer);

					if (amount_read > 0) {
						output.write(buffer, 0, amount_read);
						System.out.write(buffer, 0, amount_read);
					}
				} while (amount_read >= buffer.length);

				client.close();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
