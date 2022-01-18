
#include <orz/utils/log.h>

#include "cpuid.h"
#include "mac.h"
#include "hdserail.h"
//#include "SeetaChecker.h"
#include "SeetaAuthorizeBase.h"
#include <atomic>
#include <vector>
#include <thread>
#include "device.h"

#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif


std::vector<int> vecs;
std::atomic<long> g_total(0);
SeetaChecker* g_check = NULL;

std::atomic<long> g_errors(0);


void work_func()
{
     time_t cur = time(NULL);
     int total = vecs.size();
     int keyid = 0;
     int index = 0;
     std::string res;
     unsigned int d = 0; 

     unsigned int d2 = 0; 
     while(1)
     {
         index = rand() % total;
         keyid = vecs[index];           
         res = ""; 
         g_total++;
         d = g_total; 
         try
         {
             g_check->check(keyid, res);
             if(res.length() < 1)
             {
                 printf("----index:%u-ok----\n",d);
             }else
             {
                 d2 = g_errors++;
                 printf("----index:%u-failed----\n",d2);
             }
         }catch(...)
         {
             d2 = g_errors++;
             printf("----index:%u-failed----\n",d2);
         }
         if(time(NULL) > cur + 300)
             break;

#ifdef WIN32
        Sleep(1000);
#else
        usleep(1);
#endif
     }

     //printf("thread exit---%x\n",pthread_self());
}

int main(int argc, char **argv)
{
        std::string strdev = request_receipts();
        std::cout << "dev:" << strdev << std::endl;
        //return 0;

        vecs.push_back(3002);
        vecs.push_back(3003);
        vecs.push_back(1004);
        vecs.push_back(1002);
        vecs.push_back(1021);
        vecs.push_back(1020);
        vecs.push_back(1011);
        vecs.push_back(1006);
        vecs.push_back(1005);
        vecs.push_back(1001);
        vecs.push_back(1007);
        //vecs.push_back(1009);


	std::cout << "CPU Serial: " << get_cpu_serial() << std::endl;
	std::cout << "MAC Address: " << get_mac() << std::endl;
	std::cout << "HD Serial: " << get_hd_serial() << std::endl;

        //return 0;
       // SeetaChecker &checker = SeetaChecker::GetInstance();
       SeetaChecker checker;
       checker.init();
        std::cout << "begin check it" << std::endl;


        //return 0;
        g_check = &checker;
        std::thread threads[10];
        for(int i=0; i<10; i++)
        {
            threads[i] = std::thread(work_func);
        }

        for(int i=0; i<10; i++)
        {
            threads[i].join();
        }
        return 0;


//        std::string reason;
/*
        int n = checker.file_lock_update(reason);
        if(n == 0)
        {
           std::cout << "verify file ok" << std::endl;
        }else
        {
           std::cout << "verify file failed" << std::endl;
        }

        return 0;
*/
/*
        std::string res;
        if(argc == 2)
        {
            try
            {
                checker.check(atoi(argv[1]), res);
                std::cout << "check: func_id:" << argv[1] << ", res:" << res << std::endl;
            }catch(...)
            {
                std::cout << "check: func_id:" << argv[1] << ", exception"  << std::endl;
            }
        }
	return 0;
*/
}
