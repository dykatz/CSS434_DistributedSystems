package P1;

import java.net.*;

class Client
{
	public static void main(String[] args)
	{
		Socket s = null;

		if (args.length != 2) {
			System.err.println("err: bad args: [host] [port]");
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