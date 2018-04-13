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
import java.util.*;

public class Causal
{
	private static class Connection
	{
		public Socket socket;

		public Connection(Socket _socket)
		{
			socket = _socket;
		}

		public void close()
		{
			try {
				socket.close();
			} catch (IOException e) {
			}
		}

		public boolean closed()
		{
			return socket.isClosed();
		}
	}

	private static void realmain(int port, List<String> hosts) throws Exception
	{
		BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));

		ServerSocket server = new ServerSocket(port);
		server.setSoTimeout(1);

		List<Connection> clients = new LinkedList<Connection>();

		for (;;) {
			if (stdin.ready()) {
				String str = stdin.readLine();

				if (str == null)
					break;

				// TODO - send the user's message
			}

			try {
				Socket client_s = server.accept();
				Connection client = new Connection(client_s);

				// TODO - send a welcome message

				clients.add(client);
			} catch (SocketTimeoutException e) {
			}

			// TODO - handle inbound messages
		}

		for (Connection client : clients)
			client.socket.close();

		server.close();
	}

	public static void main(String[] args)
	{
		if (args.length < 2) {
			System.err.println("err: needs at least two arguments");
			System.err.println("usage: port hosts...");
			System.exit(1);
		}

		int port = Integer.parseInt(args[0]);
		List<String> hosts = new LinkedList<String>();

		for (int i = 1; i < args.length; ++i)
			hosts.add(args[i]);

		try {
			realmain(port, hosts);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
