#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <regex>
#include <vector>
#include <string>
#include <time.h>
#define PORT 5803

using namespace std;

struct ball{
    int dist;
    int angle; 
};

vector<ball> readballs(char arr[], int len){
    vector<ball> balls;
    string cur = "";
    int cntr = -1, ball_len = 0, ball_cntr;
    ball cur_ball;
    for(int i =0; i < len; i++){
        if(arr[i] == ' '){
            if(cntr == -1){
                ball_len = stoi(cur);
            }
            if(cntr == 0){
                cur_ball.dist = stoi(cur);
            }
            if(cntr == 1){
                cur_ball.angle = stoi(cur);
                balls.push_back(cur_ball);
                ball_cntr++;
                if(ball_cntr == ball_len) break;
            }
            cur = "";
            continue;
        }
        cur+=arr[i];
    }
    return balls;
}

int main(int argc, char const *argv[])
{
	char reqq[1024] = "pls find balls";
    	int sockfd = 0, valread;
    	struct sockaddr_in serv_addr;
   	char buffer[1024] = {0};
    	if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0){
		printf("\n Socket creation error \n");
		return -1;
   	}
   
    	serv_addr.sin_family = AF_INET;
    	serv_addr.sin_port = htons(PORT);
	serv_addr.sin_addr.s_addr = inet_addr("10.220.8.39");
	int failed = 0;
 	while(true){
	     if(sendto(sockfd, reqq, 1024, MSG_CONFIRM,
		      (const struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0){
		failed++;
     		if(failed>=10){
		       break;
		}
 		continue;		
	     }
	     memset(buffer,'\0',1024);
	     int n = recvfrom(sockfd,(char *) buffer, 1024, MSG_CONFIRM,
			     (struct sockaddr *) &serv_addr,(unsigned int *) sizeof(serv_addr));
	     vector<ball> balls = readballs(buffer,1024);
	     printf("%s \n", buffer);

	}
	printf("Failed to connect to server after 10 tries\n");	
	
    return 0;
}
