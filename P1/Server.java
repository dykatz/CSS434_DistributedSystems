package P1;

import java.io.*;
import java.net.*;
import java.nio.*;
import java.nio.channels.*;
import java.util.*;

public class Server
{
	public static void main(String args[]) throws Exception
	{
		if (args.length != 1) {
			System.err.println("err: bad args: [port]");
			System.exit(1);
		}

		Selector selector = Selector.open();

		ServerSocketChannel server = ServerSocketChannel.open();
		server.socket().bind(new InetSocketAddress(Integer.parseInt(args[0])));
		server.configureBlocking(false);
		server.register(selector, SelectionKey.OP_ACCEPT);

		HashMap<SelectionKey, String> names = new HashMap<SelectionKey, String>();
		HashSet<ObjectOutputStream> outputs = new HashSet<ObjectOutputStream>();

		for (;;) {
			int ready_channels = selector.select();

			if (ready_channels < 1)
				continue;

			Iterator<SelectionKey> keys = selector.selectedKeys().iterator();

			while (keys.hasNext()) {
				SelectionKey key = keys.next();

				if (key.isAcceptable()) {
					SocketChannel client = server.accept();
					client.configureBlocking(false);

					SelectionKey newkey = client.register(selector, SelectionKey.OP_READ);
					ObjectOutputStream newoutput = new ObjectOutputStream(client.socket().getOutputStream());
					ObjectInputStream newinput = new ObjectInputStream(client.socket().getInputStream());

					newkey.attach((Object)newinput);
					outputs.add(newoutput);
				} else if (key.isReadable()) {
					Message msg = (Message)(((ObjectInputStream)key.attachment()).readObject());

					if (msg == null)
						continue;

					msg.sender = names.getOrDefault(key, "");

					if (msg.type == Message.Type.NICK)
						names.put(key, msg.text);

					for (ObjectOutputStream output : outputs)
						output.writeObject((Object)msg);
				}
			}
		}
	}
}
