#include<stdio.h>
typedef struct{
    char name[10];
    char birth[12];
    char sex;
    char cert_tele[15];
    char tele[15];
}friend;
int main()
{
    int N;
    scanf("%d",&N);
    friend fre[N],*p;
    for(p=fre;p<(p+N);p++)
    {
        scanf("%s %s %c %s %s",p->name,p->birth,&p->sex,p->cert_tele,p->tele);
    }
    int K,Num,i;
    scanf("%d",&K);
    for(i=0;i<K;i++)
    {
        scanf("%d",&Num);
        if(Num<=N-1)
        {
            p = fre+Num;
            printf("%s %s %s %c %s\n",p->name,p->cert_tele,p->tele,p->sex,p->birth);
        }
        else printf("Not Found");
    }
    
    return 0;
}