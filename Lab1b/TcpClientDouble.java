import java.net.*;
import java.io.*;

public class TcpClientDouble
{
	public static void main(String args[]) throws Exception
	{
		if (args.length != 3) {
			System.err.println("err: needs three arguments");
			System.err.println("usage: ... port size host");
			return;
		}

		try {
			Socket socket = new Socket(args[2], Integer.parseInt(args[0]));
			ObjectOutputStream out = new ObjectOutputStream(socket.getOutputStream());
			ObjectInputStream in = new ObjectInputStream(socket.getInputStream());
			int size = Integer.parseInt(args[1]);
			Double[] data = new Double[size];

			for (int i = 0; i < data.length; ++i)
				data[i] = (double)i;

			out.writeObject((Object)data);
			data = (Double[])in.readObject();

			for (int i = 0; i < data.length; ++i)
				System.out.println(data[i]);

			socket.close();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
}
