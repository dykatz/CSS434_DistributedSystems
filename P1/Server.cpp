#include <sys/types.h>
#include <sys/event.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdio.h>
#include <err.h>

#include <memory>
#include <vector>
#include <string>

struct Client {
	int cl_conn;
	struct sockaddr_storage cl_addr;
	socklen_t cl_addrlen;
	std::string cl_addrstr;
	std::string cl_name;
	std::string cl_buffer;
};

void handle_new_connection(int, int, std::vector< std::shared_ptr<Client> >&);
void handle_new_data(int, int, int, std::vector< std::shared_ptr<Client> >&);
char *get_ip_str(const struct sockaddr*, char*, size_t);

int
main(int argc, char *argv[])
{
	int kq = kqueue();

	struct addrinfo hints, *r0;
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;
	hints.ai_flags = AI_PASSIVE;

	int error = getaddrinfo(NULL, "12345", &hints, &r0);
	int listener = -1;

	if (error)
		errx(1, "%s", gai_strerror(error));

	for (struct addrinfo *res = r0; res; res = res->ai_next) {
		listener = socket(res->ai_family, res->ai_socktype, res->ai_protocol);

		if (listener < 0)
			continue;

		if (bind(listener, res->ai_addr, res->ai_addrlen) < 0) {
			close(listener);
			listener = -1;
			continue;
		}

		if (listen(listener, 5) < 0) {
			close(listener);
			listener = -1;
			continue;
		}

		break;
	}

	freeaddrinfo(r0);

	if (listener < 0)
		err(1, "failed to start");

	struct kevent evsetter;
	EV_SET(&evsetter, listener, EVFILT_READ, EV_ADD, 0, 0, NULL);
	kevent(kq, &evsetter, 1, NULL, 0, NULL);

	std::vector< std::shared_ptr<Client> > clients;

	for (;;) {
		struct kevent evgetter[4];
		int nev = kevent(kq, NULL, 0, evgetter, 4, NULL);

		if (nev < 0)
			err(1, "kevent");

		for (int i = 0; i < nev; ++i) {
			if (evgetter[i].ident == listener) {
				handle_new_connection(kq, listener, clients);
				continue;
			}

			handle_new_data(kq, evgetter[i].ident, (int)evgetter[i].data, clients);
		}
	}
}

void
handle_new_connection(int kq, int listener, std::vector< std::shared_ptr<Client> > &clients)
{
	char buf[256];

	std::shared_ptr<Client> client = std::make_shared<Client>();
	client->cl_conn = accept(listener, (struct sockaddr*)&(client->cl_addr), &(client->cl_addrlen));

	get_ip_str((struct sockaddr*)&client->cl_addr, buf, 256);
	client->cl_addrstr = std::string(buf);

	struct kevent evsetter;
	EV_SET(&evsetter, client->cl_conn, EVFILT_READ, EV_ADD, 0, 0, (void*)clients.size());
	kevent(kq, &evsetter, 1, NULL, 0, NULL);
	clients.push_back(client);
}

void
handle_new_data(int kq, int inbound, int id, std::vector< std::shared_ptr<Client> > &clients)
{
	char buf[256];
	int nrd = read(inbound, buf, 256);
	clients[id]->cl_buffer += std::string(buf, nrd);

	for (;;) {
		int split_location = clients[id]->cl_buffer.find("\n");

		if (split_location < 0)
			break;

		std::string content = clients[id]->cl_buffer.substr(0, split_location);
		clients[id]->cl_buffer = clients[id]->cl_buffer.substr(split_location + 1);

		switch (buf[0]) {
		case 'n':
			for (int i = 0; i < clients.size(); ++i)
				dprintf(clients[i]->cl_conn, "%s (%s) is now called %s\n",
					clients[id]->cl_name.c_str(), clients[id]->cl_addrstr.c_str(), content.c_str());

			clients[id]->cl_name = content;
			break;

		case 'm':
			for (int i = 0; i < clients.size(); ++i)
				dprintf(clients[i]->cl_conn, "%s: %s\n", clients[id]->cl_name.c_str(), content.c_str());

			break;

		case 'l':
			for (int i = 0; i < clients.size(); ++i)
				dprintf(clients[id]->cl_conn, "%s (%s)\n",
					clients[i]->cl_name.c_str(), clients[i]->cl_addrstr.c_str());

			break;
		}
	}
}

char *
get_ip_str(const struct sockaddr *sa, char *s, size_t maxlen)
{
	switch (sa->sa_family) {
	case AF_INET:
		inet_ntop(AF_INET, &(((struct sockaddr_in*)sa)->sin_addr), s, maxlen);
		break;

	case AF_INET6:
		inet_ntop(AF_INET6, &(((struct sockaddr_in6*)sa)->sin6_addr), s, maxlen);
		break;

	default:
		strncpy(s, "Unknown AF", maxlen);
		return NULL;
	}

	return s;
}
