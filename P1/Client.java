package P1;

import java.util.*;
import java.net.*;
import java.nio.*;
import java.nio.channels.*;
import java.nio.file.*;
import java.io.*;

public class Client
{
	static String name;

	public static void main(String[] args) throws Exception
	{
		if (args.length != 3) {
			System.err.println("err: bad args: [host] [port] [nick]");
			System.exit(1);
		}

		name = args[2];
		int port = Integer.parseInt(args[1]);

		InetSocketAddress host = new InetSocketAddress(args[0], port);
		AsynchronousFileChannel input = AsynchronousFileChannel.open(Paths.get("/dev", "stdin"), StandardOpenOption.READ);
	}

	static Message makemsg(String text)
	{
		Message msg = new Message();
		msg.sender = name;
		msg.text = text;
		return msg;
	}
}
