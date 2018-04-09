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

package P1;

import java.io.*;
import java.net.*;

public class Client
{
	private static void realmain(String nick, String host, int port) throws Exception
	{
		Socket socket = new Socket(host, port);
		InputStream rawIn = socket.getInputStream();

		DataInputStream in = new DataInputStream(rawIn);
		DataOutputStream out = new DataOutputStream(socket.getOutputStream());
		BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));

		out.writeUTF(nick);

		for (;;) {
			if (stdin.ready()) {
				String str = stdin.readLine();

				if (str == null)
					break;

				out.writeUTF(str);
			} 

			if (rawIn.available() > 0) {
				String str = in.readUTF();
				System.out.println(str);
			}
		} 

		socket.close();
	}

	public static void main(String args[])
	{ 
		if (args.length != 3) {
			System.err.println("err: needs three arguments");
			System.err.println("usage: <nick> <host> <port>");
			System.exit(1);
		} 

		int port = Integer.parseInt(args[2]);

		try {
			realmain(args[0], args[1], port);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
} 
