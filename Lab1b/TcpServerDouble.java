import java.net.*;
import java.io.*;

public class TcpServerDouble
{
	public static void main(String args[]) throws Exception
	{
		if (args.length != 2) {
			System.err.println("err: needs two arguments");
			System.err.println("usage: ... port size");
			return;
		}

		try {
			ServerSocket svr = new ServerSocket(Integer.parseInt(args[0]));
			byte multiplier = 1;
			int size = Integer.parseInt(args[1]);

			for (;;) {
				Socket socket = svr.accept();
				ObjectInputStream in = new ObjectInputStream(socket.getInputStream());
				ObjectOutputStream out = new ObjectOutputStream(socket.getOutputStream());
				
				Double[] data = (Double[])in.readObject();

				for (int i = 0; i < data.length; ++i)
					data[i] *= multiplier;

				out.writeObject((Object)data);
				socket.close();
				multiplier *= 2;
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
}
