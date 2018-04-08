package P1;

public class Message
{
	public enum Type
	{
		MSG, NICK, JOIN, LEAVE
	}

	public Type type;
	public String sender;
	public String text;
}
