package P1;

import java.util.*;
import java.net.*;
import java.nio.*;
import java.nio.channels.*;
import java.io.*;

public class Client
{
	public static void main(String[] args)
	{
		Socket s = null;

		if (args.length != 3) {
			System.err.println("err: bad args: [host] [port] [nick]");
			System.exit(1);
		}

		try {
			s = new Socket(args[0], Integer.parseInt(args[1]));
		} catch (UnknownHostException e) {
			System.err.println("err: could not resolve host");
			System.exit(1);
		} catch (IllegalArgumentException e) {
			System.err.println("err: bad port number");
			System.exit(1);
		} catch (Exception e) {
			System.err.printf("err: failed to make socket: %s\n", e);
			System.exit(1);
		}

		try {
			s.close();
		} catch (Exception e) {
			System.err.printf("err: failed to kill socket: %s\n", e);
		}
	}
}
