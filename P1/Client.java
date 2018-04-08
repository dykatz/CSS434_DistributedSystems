package P1;

import java.io.*;
import java.lang.*;
import java.net.*;
import java.util.*;

class UserInputRunnable extends Thread
{
	Scanner in;
	ObjectOutputStream out;
	String name;

	public UserInputRunnable(Scanner _in, ObjectOutputStream _out, String _name)
	{
		in = _in;
		out = _out;
		name = _name;
	}

	public void run()
	{
		for (;;) {
			String line = in.nextLine();
			Message msg = new Message();
			msg.sender = name;

			if (line.charAt(0) == '\\') {
				int loc = line.indexOf(' ');

				if (loc >= 2) {
					String str = line.substring(1, loc);
					msg.text = line.substring(loc + 1);

					if (str.equals("name") || str.equals("nick")) {
						msg.type = Message.Type.NICK;
						name = msg.text;
					}
				}
			} else {
				msg.type = Message.Type.MSG;
				msg.text = line;
			}

			if (msg != null) {
				try {
					out.writeObject((Object)msg);
				} catch (IOException e) {
					System.err.printf("err: %s\n", e);
				}
			}
		}
	}
}

public class Client
{
	static String name;

	public static void main(String[] args) throws Exception
	{
		if (args.length != 3) {
			System.err.println("err: bad args: [host] [port] [nick]");
			System.exit(1);
		}

		Socket sock = new Socket(args[0], Integer.parseInt(args[1]));
		ObjectOutputStream out = new ObjectOutputStream(sock.getOutputStream());
		ObjectInputStream in = new ObjectInputStream(sock.getInputStream());
		Scanner user = new Scanner(System.in);
		UserInputRunnable bgthread = new UserInputRunnable(user, out, args[2]);
		bgthread.start();

		for (;;) {
			Message inbound = (Message)in.readObject();

			switch (inbound.type) {
			case MSG:
				System.out.printf("<%s> %s\n", inbound.sender, inbound.text);
				break;
			case NICK:
				System.out.printf("%s is now called %s\n", inbound.sender, inbound.text);
				break;
			case JOIN:
				System.out.printf("%s has joined the server\n", inbound.sender);
				break;
			case LEAVE:
				System.out.printf("%s has left the server\n", inbound.sender);
				break;
			}
		}
	}
}
