//
// Created by chen on 2020/1/21.
//

#ifndef NIGHTSIGHT_UTILS_H
#define NIGHTSIGHT_UTILS_H

#endif //NIGHTSIGHT_UTILS_H


#include <ctime>
void Delay(int time){ //time*1000為秒數
    clock_t   now   =   clock();

    while( clock() -  now  <  time*1000);
}

