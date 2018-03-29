package P1;

import java.util.*;
import java.net.*;
import java.nio.*;
import java.nio.channels.*;
import java.io.*;

class ClientRef
{
	public List<ClientRef> others;
	public AsynchronousSocketChannel channel;
	public String name;
	public ByteBuffer buffer;

	public ClientRef(List<ClientRef> oth, AsynchronousSocketChannel ch)
	{
		others = oth;
		channel = ch;
		buffer = ByteBuffer.allocate(256);

		channel.read(buffer, null, new CompletionHandler<Integer,Void>()
		{
			public void completed(Integer len, Void att)
			{
				try {
					ByteArrayInputStream bais = new ByteArrayInputStream(
						buffer.array(), 0, buffer.limit());
					ObjectInputStream ois = new ObjectInputStream(bais);
					handle((Message)ois.readObject());
				} catch (Exception e) {
					System.err.printf("err: failed to recv message: %s\n", e);
				}

				buffer.clear();
				channel.read(buffer, null, this);
			}

			public void failed(Throwable exc, Void att)
			{
				System.err.println(exc);
			}
		});
	}

	void handle(Message msg)
	{
		if (msg.type == Message.Type.NAME)
			name = msg.text;

		for (ClientRef ref : others)
			ref.send(msg);
	}

	void send(Message msg)
	{
		try {
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			ObjectOutputStream oos = new ObjectOutputStream(baos);
			oos.writeObject(msg);
			oos.close();
			ByteBuffer bb = ByteBuffer.wrap(baos.toByteArray());

			while (bb.hasRemaining())
				channel.write(bb);
		} catch (IOException e) {
			System.err.printf("err: failed to send message: %s\n", e);
		}
	}
}

public class Server
{
	public static void main(String[] args)
	{
		if (args.length != 1) {
			System.err.println("err: 1 arg required: [port]");
			System.exit(1);
		}

		int port = Integer.parseInt(args[0]);

		try {
			serve(port);
		} catch (IOException e) {
			System.err.printf("err: server broke: %s\n", e);
		}

		Thread.sleep(Long.MAX_VALUE);
	}

	static void serve(int port) throws IOException
	{
		InetSocketAddress host = new InetSocketAddress(port);
		AsynchronousServerSocketChannel srv =
			AsynchronousServerSocketChannel.open().bind(host);
		List<ClientRef> clients = new ArrayList<ClientRef>();

		srv.accept(null, new CompletionHandler<AsynchronousSocketChannel,Void>()
		{
			public void completed(AsynchronousSocketChannel ch, Void att)
			{
				new ClientRef(clients, ch);
				srv.accept(null, this);
			}

			public void failed(Throwable exc, Void att)
			{
				System.err.println(exc);
			}
		});
	}
}
