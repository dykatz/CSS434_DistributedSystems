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

public class Server
{
	private static class Connection
	{
		public String nick;
		private Socket socket;
		private InputStream rawIn;
		private DataInputStream in;
		private DataOutputStream out;

		public Connection(Socket _socket) throws IOException
		{
			socket = _socket;
			rawIn = socket.getInputStream();
			in = new DataInputStream(rawIn);
			out = new DataOutputStream(socket.getOutputStream());

			nick = in.readUTF();
		}

		public boolean available()
		{
			try {
				return rawIn.available() > 0;
			} catch (IOException e) {
				try {
					socket.close();
				} catch (IOException f) {
				}

				return false;
			}
		}

		public String read()
		{
			try {
				return in.readUTF();
			} catch (IOException e) {
				try {
					socket.close();
				} catch (IOException f) {
				}

				return null;
			}
		}

		public void write(String _out)
		{
			try {
				out.writeUTF(_out);
			} catch (IOException e) {
				try {
					socket.close();
				} catch (IOException f) {
				}
			}
		}

		public boolean closed()
		{
			return socket.isClosed();
		}
	}

	private static void realmain(int port) throws Exception
	{
		List<Connection> clients = new LinkedList<Connection>();
		ServerSocket socket = new ServerSocket(port);
		socket.setSoTimeout(1);

		for (;;) {
			try {
				Socket news = socket.accept();
				Connection client = new Connection(news);

				for (Connection client2 : clients)
					client2.write(client.nick + " has joined");

				clients.add(client);
			} catch (SocketTimeoutException e) {
			}

			for (Connection client : clients) {
				if (client.available()) {
					String data = "<" + client.nick + "> " + client.read();

					for (Connection client2 : clients)
						client2.write(data);
				}
			}

			for (Iterator<Connection> i = clients.iterator(); i.hasNext(); ) {
				Connection client = i.next();

				if (client.closed()) {
					i.remove();

					for (Connection client2 : clients)
						client2.write(client.nick + " has left");

					continue;
				}
			}
		}
	}

	public static void main(String args[])
	{
		if (args.length != 1) {
			System.err.println("err: needs one argument");
			System.exit(1);
		}

		int port = Integer.parseInt(args[0]);

		try {
			realmain(port);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
