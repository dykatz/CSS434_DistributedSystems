package P1;

import java.net.*;

class Server
{
	public static void main(String[] args)
	{
		ServerSocket s = null;

		if (args.length != 1) {
			System.err.println("err: bad args: [port]");
			System.exit(1);
		}

		try {
			s = new ServerSocket(Integer.parseInt(args[0]));
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
