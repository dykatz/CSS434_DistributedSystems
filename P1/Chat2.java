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

public class Chat2
{
	private static class Message implements Serializable
	{
		private int stamp;
		private String text;
		private String from;

		public Message(int _s, String _t, String _f)
		{
			stamp = _s;
			text = _t;
			from = _f;
		}

		public int getStamp()
		{
			return stamp;
		}

		public String toString()
		{
			return from + ": " + text;
		}

		public String getFrom()
		{
			return from;
		}
	}

	private static class MessageComparator implements Comparator<Message>
	{
		public int compare(Message a, Message b)
		{
			int sa = a.getStamp(), sb = b.getStamp();

			if (sa < sb)
				return -1;
			else if (sa > sb)
				return 1;
			else
				return a.toString().compareTo(b.toString());
		}

		public boolean equals(Message a, Message b)
		{
			int sa = a.getStamp(), sb = b.getStamp();

			if (sa != sb)
				return false;
			else
				return a.toString().equals(b.toString());
		}
	}

	private static class Connection
	{
		private Set<Connection> container;
		private String name;
		private Socket socket;
		private InputStream rawIn;
		private OutputStream rawOut;
		private ObjectInputStream in;
		private ObjectOutputStream out;

		public Connection(Set<Connection> _c, Socket _s) throws Exception
		{	
			container = _c;
			socket = _s;
			rawIn = _s.getInputStream();
			rawOut = _s.getOutputStream();

			container.add(this);
		}

		public void onAccept(String _n, int _s)
		{
			try {
				in = new ObjectInputStream(rawIn);
				out = new ObjectOutputStream(rawOut);

				name = (String)in.readObject();
				out.writeObject((Object)new Message(_s, null, _n));
			} catch (Exception e) {
				close();
			}
		}

		public int onStartup(String _n)
		{
			try {
				out = new ObjectOutputStream(rawOut);
				in = new ObjectInputStream(rawIn);

				out.writeObject((Object)_n);
				Message msg = read();
				name = msg.getFrom();
				return msg.getStamp();
			} catch (Exception e) {
				close();
				return 0;
			}
		}

		public boolean isAvailable()
		{
			try {
				return rawIn.available() > 0;
			} catch (IOException e) {
				close();
				return false;
			}
		}

		public void close()
		{
			container.remove(this);
			try { out.close();    } catch (IOException e) {}
			try { in.close();     } catch (IOException e) {}
			try { rawOut.close(); } catch (IOException e) {}
			try { rawIn.close();  } catch (IOException e) {}
			try { socket.close(); } catch (IOException e) {}
		}

		public Message read()
		{
			try {
				return (Message)in.readObject();
			} catch (Exception e) {
				e.printStackTrace();
				close();
				return null;
			}
		}

		public void write(Message _m)
		{
			try {
				out.writeObject((Object)_m);
				out.flush();
			} catch (Exception e) {
				close();
			}
		}
	}

	private static int stamp = 0;
	private static Set<Connection> connections;
	private static NavigableSet<Message> backlog;
	private static BufferedReader stdin;
	private static ServerSocket listener;
	private static String nick;

	public static void main(String[] args) throws Exception
	{
		if (args.length % 2 > 0 || args.length < 2) {
			System.err.println("err: wrong number of arguments");
			System.err.println("usage: nick port [host0 port0 host1 port1...]");
			System.exit(1);
		}

		connections = new HashSet<Connection>();
		backlog = new TreeSet<Message>(new MessageComparator());
		stdin = new BufferedReader(new InputStreamReader(System.in));
		listener = new ServerSocket(Integer.parseInt(args[1]));
		nick = args[0];

		for (int i = 2; i < args.length - 1; i += 2) {
			String host = args[i];
			int port = Integer.parseInt(args[i + 1]);

			try {
				Socket socket = new Socket(host, port);
				Connection client = new Connection(connections, socket);
				stamp = Math.max(stamp, client.onStartup(nick));
				System.out.printf("successfully connected to %s on %d!\n", host, port);
			} catch (Exception e) {
				System.err.printf("warning: failed to connect to %s on %d at startup\n", host, port);
			}
		}

		listener.setSoTimeout(1);

		for (;;) {
			try {
				Socket socket = listener.accept();
				Connection client = new Connection(connections, socket);
				client.onAccept(nick, stamp);
			} catch (SocketTimeoutException e) {}

			if (stdin.ready()) {
				Message input = new Message(stamp, stdin.readLine(), nick);
				++stamp;

				for (Connection client : connections)
					client.write(input);
			}

			for (Connection client : connections) {
				if (client.isAvailable()) {
					Message input = client.read();

					if (input.getStamp() > stamp) {
						backlog.add(input);
					} else {
						System.out.println(input);
						stamp = Math.max(stamp, input.getStamp() + 1);

						for (Message msg : backlog) {
							if (msg.getStamp() <= stamp) {
								System.out.println(msg);
								stamp = Math.max(stamp, msg.getStamp() + 1);
								backlog.remove(msg);
							}
						}
					}
				}
			}
		}
	}
}
