#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <regex>
#include <vector>
#include <string>
#define PORT 8888

using namespace std;

struct ball{
    int dist;
    int angle; 
};

vector<ball> readballs(char[] arr, int len){
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
    int sockfd = 0, valread;
    struct sockaddr_in serv_addr;
    char buffer[1024] = {0};
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        return -1;
    }
   
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
       
    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET, "10.220.8.28", &serv_addr.sin_addr)<=0) 
    {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }
   
    if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    {
        printf("\nConnection Failed \n");
        return -1;
    }
    
    valread = read( sockfd , buffer, 1024);
    while(valread){
        vector<ball> balls = readballs(buffer,1024);
        printf("%s\n",buffer );
        memset(buffer,'\0',1024);
        valread = read(sockfd,buffer,1024);
    }
    return 0;
}