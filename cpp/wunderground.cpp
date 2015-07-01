#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <rapidxml.hpp>

using namespace std;
using namespace rapidxml;

int main(void){

  //pull from Wunderground
  int ret = system("wget http://api.wunderground.com/api/eea590fdddcc01bb/conditions/q/pws:KOKCYRIL3.xml");
  return 0;
}
